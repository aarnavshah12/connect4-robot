"""End-to-end plumbing test for vision.py: synthetic camera -> shared memory
-> worker process -> detection queue. No real camera, no model."""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.fake_worker import fake_worker
from vision import Vision


class SyntheticCamera(threading.Thread):
    """Same interface as vision.Camera, no cv2.VideoCapture."""

    def __init__(self, shape=(48, 64, 3)):
        super().__init__(daemon=True)
        self.shape = shape
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._stop = threading.Event()
        self.fps = 30.0

    def run(self):
        while not self._stop.is_set():
            frame = np.zeros(self.shape, dtype=np.uint8)
            frame[0, 0, 0] = (self._frame_id + 1) % 200
            with self._lock:
                self._frame = frame
                self._frame_id += 1
            time.sleep(0.02)

    def read(self):
        with self._lock:
            if self._frame is None:
                return 0, None
            return self._frame_id, self._frame.copy()

    def stop(self):
        self._stop.set()


def test_frames_flow_to_worker_and_detections_flow_back(tmp_path):
    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps({
        "homography": np.eye(3).tolist(), "cell_w": 8, "cell_h": 8,
        "crop": [0, 0, 64, 48], "frame_size": [64, 48],
    }))
    cam = SyntheticCamera()
    v = Vision({"model_id": "x/y/1", "camera_index": 0},
               calib_path=calib, worker=fake_worker, camera=cam)
    v.start()
    try:
        deadline = time.time() + 10
        got = (0, [], 0.0)
        while time.time() < deadline:
            got = v.detections()
            if got[1]:
                break
            time.sleep(0.05)
        fid, dets, fps = got
        assert dets, "no detections arrived from the worker process"
        x, y, w, h, cls, conf = dets[0]
        assert cls == "red" and conf == 0.95
        assert x > 10, "worker never saw the camera's pixel data"
        # newest-wins: keep polling, frame id should advance
        time.sleep(0.3)
        fid2, _, _ = v.detections()
        assert fid2 > fid
        # draw-loop side sees frames too
        fid3, frame = v.frame()
        assert frame is not None and frame.shape == (48, 64, 3)
    finally:
        v.stop()
        cam.stop()
