"""A tiny CUDA stream simulator for testing asynchronous pipelines on CPU.

Nothing executes when it is enqueued.  Work runs only when something
*synchronises* -- the worst case of a GPU lagging far behind the host -- and
every copy reads its source at *execution* time.  Host buffers that are
refilled too early, device buffers that are overwritten while queued kernels
still need them, and missing stream dependencies therefore all produce wrong
results, exactly as they (sometimes) do on real hardware.
"""

import threading

import numpy as np

_LOCK = threading.RLock()
_TLS = threading.local()


class Event:
    def __init__(self):
        self.done = False

    def synchronize(self):
        _run_until(lambda: self.done)


class Stream:
    all_streams = []

    def __init__(self, non_blocking=True):
        self.queue = []
        with _LOCK:
            Stream.all_streams.append(self)

    # -- enqueue ----------------------------------------------------------
    def enqueue(self, fn, needs=None):
        with _LOCK:
            self.queue.append((fn, needs))

    def record(self):
        ev = Event()
        self.enqueue(lambda: setattr(ev, "done", True))
        return ev

    def wait_event(self, event):
        self.enqueue(lambda: None, needs=event)

    def synchronize(self):
        _run_until(lambda: not self.queue)

    # -- "current stream" handling (thread local, like cupy) ---------------
    def __enter__(self):
        stack = getattr(_TLS, "stack", None)
        if stack is None:
            stack = _TLS.stack = []
        stack.append(self)
        return self

    def __exit__(self, *exc):
        _TLS.stack.pop()
        return False


DEFAULT = Stream()


def current_stream():
    stack = getattr(_TLS, "stack", None)
    return stack[-1] if stack else DEFAULT


_RNG = np.random.default_rng(0)


def seed(value):
    """Re-seed the adversarial scheduler (explore another interleaving)."""
    global _RNG
    _RNG = np.random.default_rng(value)


def _step():
    """Run ONE operation of a randomly chosen runnable stream: streams are
    independent on real hardware, so every interleaving that respects the
    recorded dependencies is legal."""
    runnable = [
        s for s in Stream.all_streams
        if s.queue and (s.queue[0][1] is None or s.queue[0][1].done)
    ]
    if not runnable:
        return False
    stream = runnable[int(_RNG.integers(len(runnable)))]
    fn, _needs = stream.queue.pop(0)
    fn()
    return True


def _run_until(done):
    with _LOCK:
        while not done():
            if not _step():
                raise RuntimeError("stream deadlock: nothing can make progress")


def synchronize_all():
    _run_until(lambda: all(not s.queue for s in Stream.all_streams))


class DeviceArray:
    def __init__(self, shape, dtype):
        self.data = np.zeros(shape, dtype=dtype)
        self.shape, self.dtype = self.data.shape, self.data.dtype

    def set(self, host):
        # asynchronous copy: reads `host` when the copy *executes*
        current_stream().enqueue(lambda: np.copyto(self.data, host))


class _Device:
    def __init__(self, _id):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Cuda:
    Stream = Stream
    Device = _Device

    @staticmethod
    def get_current_stream():
        return current_stream()


class SimModule:
    cuda = _Cuda


class SimBackend:
    """Duck-typed Backend for RowSpaceMapLoader."""

    name = "cupy"
    device_id = 0
    module = SimModule

    def create_stream(self, non_blocking=True):
        return Stream()

    def alloc_pinned(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return DeviceArray(shape, dtype)


def enqueue_kernel(dev_array, sink):
    """A 'kernel' on the current stream: reads the device buffer when it
    executes and appends a copy to ``sink``."""
    current_stream().enqueue(lambda: sink.append(dev_array.data.copy()))
