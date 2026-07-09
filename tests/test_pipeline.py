"""Tests for the PinnedMapPipeline double-buffered upload helper."""

import unittest
from types import SimpleNamespace

import numpy as np

from CosmoFuse.backend import get_backend
from CosmoFuse.pipeline import PinnedMapPipeline


class TestPinnedMapPipelineCPU(unittest.TestCase):
    def _make_corr_stub(self):
        return SimpleNamespace(backend=get_backend("cpu"), map_dtype=np.dtype(np.float64))

    def test_stage_wait_round_trip(self):
        corr = self._make_corr_stub()
        shapes = {"shear": (2, 2, 8), "w": (2, 8)}
        rng = np.random.default_rng(42)
        maps = [
            {"shear": rng.random((2, 2, 8)), "w": rng.random((2, 8))}
            for _ in range(3)
        ]

        pipe = PinnedMapPipeline(corr, shapes)
        dev = pipe.wait(pipe.stage(maps[0]))
        np.testing.assert_array_equal(dev["shear"], maps[0]["shear"])
        np.testing.assert_array_equal(dev["w"], maps[0]["w"])

        # Staging map 1 must not disturb the currently-visible slot.
        nxt = pipe.stage(maps[1])
        np.testing.assert_array_equal(dev["shear"], maps[0]["shear"])
        dev = pipe.wait(nxt)
        np.testing.assert_array_equal(dev["shear"], maps[1]["shear"])
        np.testing.assert_array_equal(dev["w"], maps[1]["w"])

        # wait(None) keeps the current slot (last map of the loop).
        same = pipe.wait(None)
        np.testing.assert_array_equal(same["shear"], maps[1]["shear"])

        dev = pipe.wait(pipe.stage(maps[2]))
        np.testing.assert_array_equal(dev["w"], maps[2]["w"])

    def test_dtype_override(self):
        corr = self._make_corr_stub()
        pipe = PinnedMapPipeline(corr, {"w": (4,)}, dtype=np.float32)
        dev = pipe.wait(pipe.stage({"w": np.arange(4, dtype=np.float64)}))
        self.assertEqual(dev["w"].dtype, np.float32)
        np.testing.assert_array_equal(dev["w"], np.arange(4, dtype=np.float32))


class TestPinnedMapPipelineStreamContract(unittest.TestCase):
    """The CUDA-side contract, driven with fakes: uploads are enqueued on
    the pipeline stream and the compute stream waits on the copy event
    exactly when a staged slot is swapped in."""

    def _make_fake_gpu_corr(self):
        events = []

        class FakeEvent:
            def __init__(self, ident):
                self.ident = ident

        class FakeStream:
            def __init__(self):
                self.entered = 0
                self.recorded = 0

            def record(self):
                self.recorded += 1
                return FakeEvent(self.recorded)

            def __enter__(self):
                self.entered += 1
                return self

            def __exit__(self, *_exc):
                return False

        class FakeCurrentStream:
            def wait_event(self, event):
                events.append(("wait", event.ident))

        class FakeDeviceArray:
            def __init__(self, shape, dtype):
                self.data = np.zeros(shape, dtype=dtype)
                self.dtype = self.data.dtype

            def set(self, host):
                self.data[...] = host

        upload_stream = FakeStream()
        current_stream = FakeCurrentStream()

        class FakeBackend:
            module = SimpleNamespace(
                cuda=SimpleNamespace(get_current_stream=lambda: current_stream)
            )

            @staticmethod
            def create_stream(non_blocking=True):
                return upload_stream

            @staticmethod
            def alloc_pinned(shape, dtype):
                return np.empty(shape, dtype=dtype)

            @staticmethod
            def zeros(shape, dtype):
                return FakeDeviceArray(shape, dtype)

        corr = SimpleNamespace(backend=FakeBackend(), map_dtype=np.dtype(np.float64))
        return corr, upload_stream, events

    def test_uploads_use_stream_and_wait_on_swap(self):
        corr, upload_stream, events = self._make_fake_gpu_corr()
        pipe = PinnedMapPipeline(corr, {"w": (3,)})

        token = pipe.stage({"w": np.array([1.0, 2.0, 3.0])})
        # One .set() inside the stream context, one recorded copy event.
        self.assertEqual(upload_stream.entered, 1)
        self.assertEqual(upload_stream.recorded, 1)
        self.assertEqual(events, [])  # nothing waits until the swap

        dev = pipe.wait(token)
        self.assertEqual(events, [("wait", 1)])
        np.testing.assert_array_equal(dev["w"].data, [1.0, 2.0, 3.0])

        # wait(None) must not wait again.
        pipe.wait(None)
        self.assertEqual(events, [("wait", 1)])


if __name__ == "__main__":
    unittest.main()
