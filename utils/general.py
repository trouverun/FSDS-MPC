import numpy as np
from queue import Empty


def sm_array(shared_mem, shape, dtype=np.float32):
    return np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf)


def flush_queue(queue):
    try:
        while True:
            queue.get_nowait()
    except Empty:
        return