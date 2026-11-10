"""Vision pipeline: camera capture thread + model inference in its own
PROCESS (plan rule: process, not thread — the GIL must never let model
compute stutter the draw loop).

Data flow:
    Camera thread  -> latest frame -> shared memory (+ counter)
                                   -> .frame() for the draw loop
    Worker process -> copies newest frame, CROPS to the calibrated board box,
                      runs the model on MPS, maps boxes back to full-frame
                      pixels, publishes to a queue (newest wins)
    Vision.detections() -> newest (frame_id, dets, infer_fps) without blocking

dets are (x, y, w, h, class_name, confidence) in FULL-FRAME pixels; board.py
consumes the (x, y) centroids, overlay.py draws the boxes.
For the cell classifier, dets=None means an uncertain/occluded scan.
"""

import json
import multiprocessing as mp
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np


def _open_capture(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)  # prefer 60; floor is 30 (plan)
    return cap


class Camera(threading.Thread):
    """Grabs frames as fast as the camera gives them; keeps only the newest."""

    def __init__(self, index=0, capture_factory=_open_capture):
        super().__init__(daemon=True)
        self.index = index
        self._factory = capture_factory
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._stop = threading.Event()
        self.fps = 0.0

    def run(self):
        cap = self._factory(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.index} did not open")
        # lock exposure/WB after a short warmup so detections don't flicker
        t_warm = time.time() + 1.5
        locked = False
        last = time.time()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if not locked and time.time() > t_warm:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                locked = True
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
            last = now
            with self._lock:
                self._frame = frame
                self._frame_id += 1
        cap.release()

    def read(self):
        """Newest (frame_id, frame) or (0, None) before first capture."""
        with self._lock:
            if self._frame is None:
                return 0, None
            return self._frame_id, self._frame.copy()

    def stop(self):
        self._stop.set()


def _extract_predictions(result, crop_xy):
    """inference-package result -> [(x, y, w, h, class_name, conf)] full-frame."""
    ox, oy = crop_xy
    out = []
    preds = getattr(result, "predictions", None) or []
    for p in preds:
        if isinstance(p, dict):
            x, y = p["x"], p["y"]
            w, h = p["width"], p["height"]
            cls = p.get("class") or p.get("class_name")
            conf = p["confidence"]
        else:
            x, y, w, h = p.x, p.y, p.width, p.height
            cls = getattr(p, "class_name", None) or getattr(p, "class_", None)
            conf = p.confidence
        out.append((x + ox, y + oy, w, h, str(cls).lower(), conf))
    return out


def inference_worker(shm_name, shape, counter, crop, model_id, api_key, out_q, stop_evt):
    """Runs in a separate process. Loads the model, then: newest frame ->
    crop -> infer -> publish. Never blocks the producer side."""
    import torch  # imported here: the parent process must stay torch-free

    mps = torch.backends.mps.is_available()
    print(f"[vision] torch {torch.__version__}, mps available: {mps}"
          + ("" if mps else "  <-- WARNING: CPU inference, expect low fps"))

    from inference import get_model

    model = get_model(model_id=model_id, api_key=api_key)
    print(f"[vision] model {model_id} loaded ({type(model).__name__})")

    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    x0, y0, x1, y1 = crop
    seen = 0
    last_t = time.time()
    fps = 0.0
    try:
        while not stop_evt.is_set():
            if counter.value == seen:
                time.sleep(0.002)
                continue
            seen = counter.value
            frame = buf[y0:y1, x0:x1].copy()
            try:
                result = model.infer(frame)[0]
            except Exception as e:  # keep the pipeline alive on a bad frame
                print(f"[vision] infer error: {e}")
                continue
            now = time.time()
            fps = 0.8 * fps + 0.2 / max(now - last_t, 1e-6)
            last_t = now
            dets = _extract_predictions(result, (x0, y0))
            try:
                out_q.put_nowait((seen, dets, round(fps, 1)))
            except Exception:
                try:
                    out_q.get_nowait()
                    out_q.put_nowait((seen, dets, round(fps, 1)))
                except Exception:
                    pass
    finally:
        shm.close()


def torch_inference_worker(shm_name, shape, counter, crop, config, calib_path, out_q, stop_evt):
    """Batch all 42 calibrated cell crops in one PyTorch inference call."""
    from board import Calibration
    from ml.cells import CellReader
    from ml.utils import choose_device

    device = choose_device(config.get("vision_device", "auto"))
    reader = CellReader(config["vision_checkpoint"], Calibration.load(calib_path), device,
                        config.get("vision_min_confidence", 0.9))
    print(f"[vision] PyTorch cell classifier loaded on {device}", flush=True)
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    seen, fps, last_t = 0, 0.0, time.monotonic()
    try:
        while not stop_evt.is_set():
            with counter.get_lock():
                fid = counter.value
                frame = buf.copy() if fid != seen else None
            if frame is None:
                time.sleep(0.003)
                continue
            seen = fid
            try:
                dets = reader.predict(frame)
            except Exception as exc:
                print(f"[vision] unusable frame: {exc}", flush=True)
                dets = None
            now = time.monotonic()
            fps = 0.8 * fps + 0.2 / max(now - last_t, 1e-6)
            last_t = now
            try:
                out_q.put_nowait((fid, dets, round(fps, 1)))
            except Exception:
                try:
                    out_q.get_nowait()
                    out_q.put_nowait((fid, dets, round(fps, 1)))
                except Exception:
                    pass
    finally:
        shm.close()


class Vision:
    """Facade the rest of the app talks to."""

    def __init__(self, config, calib_path=None, worker=None, camera=None):
        self.cfg = dict(config)
        self._worker_fn = worker
        calib_path = calib_path or Path(__file__).parent / "calibration.json"
        self.calib_path = str(Path(calib_path).resolve())
        self.backend = self.cfg.get("vision_backend", "roboflow")
        if self.backend not in ("torch", "roboflow"):
            raise ValueError(f"unknown vision_backend: {self.backend}")
        if self.backend == "torch" and worker is None:
            checkpoint = Path(self.cfg.get("vision_checkpoint", "artifacts/vision/best.pt"))
            if not checkpoint.is_absolute():
                checkpoint = Path(__file__).parent / checkpoint
            if not checkpoint.is_file():
                raise FileNotFoundError(f"vision checkpoint missing: {checkpoint}; train with ml.train_vision first")
            self.cfg["vision_checkpoint"] = str(checkpoint)
        calib = json.loads(Path(calib_path).read_text())
        self.crop = calib.get("crop", [0, 0, 1280, 720])
        w, h = calib.get("frame_size", [1280, 720])
        self.shape = (h, w, 3)

        self.camera = camera if camera is not None else Camera(config.get("camera_index", 0))
        self._shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(self.shape)))
        self._shm_buf = np.ndarray(self.shape, dtype=np.uint8, buffer=self._shm.buf)
        self._counter = mp.Value("Q", 0)
        self._stop_evt = mp.Event()
        self._queue = mp.Queue(maxsize=2)
        self._proc = None
        self._pub_thread = None
        self._latest_dets = (0, None, 0.0)

    def start(self):
        if self._worker_fn is not None:
            target = self._worker_fn
            model_args = (self.cfg.get("model_id", ""), self.cfg.get("roboflow_api_key", ""))
        elif self.backend == "torch":
            target, model_args = torch_inference_worker, (self.cfg, self.calib_path)
        else:
            target = inference_worker
            model_args = (self.cfg["model_id"], self.cfg.get("roboflow_api_key", ""))
        self.camera.start()
        self._proc = mp.Process(
            target=target,
            args=(self._shm.name, self.shape, self._counter, tuple(self.crop),
                  *model_args, self._queue, self._stop_evt),
            daemon=True,
        )
        self._proc.start()
        self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._pub_thread.start()

    def _publish_loop(self):
        """Copy the newest camera frame into shared memory for the worker."""
        seen = 0
        while not self._stop_evt.is_set():
            fid, frame = self.camera.read()
            if frame is None or fid == seen:
                time.sleep(0.003)
                continue
            seen = fid
            if frame.shape != self.shape:  # camera negotiated another size
                frame = cv2.resize(frame, (self.shape[1], self.shape[0]))
            with self._counter.get_lock():
                self._shm_buf[:] = frame
                self._counter.value = fid

    def frame(self):
        """Newest camera frame for the draw loop: (frame_id, ndarray|None)."""
        return self.camera.read()

    def detections(self):
        """Newest (frame_id, dets, infer_fps); never blocks."""
        if (self.backend == "torch" and self._proc is not None
                and self._proc.exitcode is not None and not self._stop_evt.is_set()):
            raise RuntimeError(f"vision worker exited with code {self._proc.exitcode}; see its startup error")
        while True:
            try:
                self._latest_dets = self._queue.get_nowait()
            except Exception:
                break
        return self._latest_dets

    def stop(self):
        self._stop_evt.set()
        self.camera.stop()
        if self._pub_thread:
            self._pub_thread.join(timeout=2)
        if self._proc:
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2)
        self._shm.close()
        self._shm.unlink()
