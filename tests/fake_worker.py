"""Test-only inference worker: same process/shm/queue plumbing as the real
one, but publishes a fixed detection instead of running a model.
Top-level module so multiprocessing spawn can import it in the child."""

import time

import numpy as np
from multiprocessing import shared_memory


def fake_worker(shm_name, shape, counter, crop, model_id, api_key, out_q, stop_evt):
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    seen = 0
    try:
        while not stop_evt.is_set():
            if counter.value == seen:
                time.sleep(0.002)
                continue
            seen = counter.value
            # prove we actually read the frame the camera wrote
            checksum = int(buf[0, 0, 0])
            dets = [(10.0 + checksum, 10.0, 6.0, 6.0, "red", 0.95)]
            try:
                out_q.put_nowait((seen, dets, 20.0))
            except Exception:
                try:
                    out_q.get_nowait()
                    out_q.put_nowait((seen, dets, 20.0))
                except Exception:
                    pass
    finally:
        shm.close()
