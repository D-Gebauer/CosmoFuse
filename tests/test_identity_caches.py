"""Caches that used to be keyed on a recycled address.

``id()`` and a CUDA pool pointer identify an object only while it is alive.
Three caches keyed on one or the other, and none of them held a reference to
what it had keyed, so a *different* object at the same address was taken for
the cached one and the library returned the previous call's answer:

- the aperture filter (``PairGeometry.aperture_filter_key``)
- device weight maps (``Correlation._fingerprint_weights``)
- frozen host weight maps (``_FINGERPRINT_MEMO``)

These gates hold the addresses deliberately recycled, which is the only way
to make the failure deterministic.
"""

import gc
import unittest

import numpy as np

from CosmoFuse.correlation_helpers import Q_crittenden, Q_schneider
from CosmoFuse.pair_geometry import PairGeometry
from CosmoFuse.utils import live_object_serial


class TestLiveObjectSerial(unittest.TestCase):
    def test_recycled_addresses_get_distinct_serials(self):
        # CPython reuses a small object's address as soon as it is freed, so
        # no gc.collect() is needed to force the collision this guards.
        n = 200
        serials, addresses = set(), set()
        for _ in range(n):
            obj = np.zeros(3)
            addresses.add(id(obj))
            serials.add(live_object_serial(obj))
            del obj
        # the point of the test: the addresses really were reused
        self.assertLess(len(addresses), len(serials))
        self.assertEqual(len(serials), n)

    def test_stable_while_the_object_is_alive(self):
        obj = np.zeros(3)
        self.assertEqual(live_object_serial(obj), live_object_serial(obj))

    def test_keeps_nothing_alive(self):
        from CosmoFuse.utils import _OBJECT_SERIALS

        obj = np.zeros(3)
        live_object_serial(obj)
        key = id(obj)
        self.assertIn(key, _OBJECT_SERIALS)
        del obj
        gc.collect()
        self.assertNotIn(key, _OBJECT_SERIALS)

    def test_objects_that_cannot_be_weak_referenced_fall_back(self):
        # ints are not weak-referenceable; the address is still returned
        self.assertEqual(live_object_serial(12345), id(12345))


class TestApertureFilterKey(unittest.TestCase):
    def test_transient_filters_at_the_same_address_differ(self):
        keys, addresses = [], []
        for scale in (1.0, 2.0):
            fn = lambda theta, theta_Q, _s=scale: _s * np.asarray(theta)
            addresses.append(id(fn))
            keys.append(PairGeometry.aperture_filter_key(fn))
            del fn
            gc.collect()
        self.assertEqual(addresses[0], addresses[1], "addresses were not reused")
        self.assertNotEqual(keys[0], keys[1])

    def test_same_filter_still_hits(self):
        fn = lambda theta, theta_Q: np.asarray(theta)
        self.assertEqual(
            PairGeometry.aperture_filter_key(fn),
            PairGeometry.aperture_filter_key(fn),
        )

    def test_the_default_filter_keeps_its_stable_name(self):
        self.assertEqual(
            PairGeometry.aperture_filter_key(Q_crittenden), "Q_crittenden"
        )
        self.assertNotEqual(
            PairGeometry.aperture_filter_key(Q_schneider), "Q_crittenden"
        )


def _small_cpu_corr():
    from CosmoFuse.correlations import Correlation

    corr = Correlation(
        nside=1,
        phi_center=np.array([0.0]),
        theta_center=np.array([0.0]),
        nbins=1,
        theta_min=1.0,
        theta_max=2.0,
        patch_size=1.0,
        theta_Q=1.0,
        device="cpu",
    )
    corr.pair_inds = [np.array([[0, 1], [1, 2]], dtype=np.uint32)]
    corr.pair_exp2phi = [np.ones((2, 2), dtype=np.complex128)]
    corr.bins = [np.array([2], dtype=np.uint32)]
    corr.Q_inds = [np.array([0], dtype=np.uint32)]
    corr.Q_cos = [np.array([1.0], dtype=np.float64)]
    corr.Q_sin = [np.array([0.0], dtype=np.float64)]
    corr.Q_val = [np.array([1.0], dtype=np.float64)]
    corr.Q_patch_area = [1.0]
    corr.prepare()
    return corr


class TestFrozenWeightFingerprint(unittest.TestCase):
    """``_FINGERPRINT_MEMO`` keys a blake2b digest on ``(id, data pointer)``.

    The docstring recommends freezing weight maps to skip rehashing, so the
    intended usage is exactly the one that breaks: build one frozen map per
    realisation and drop it, and the next lands on the recycled address.
    """

    def setUp(self):
        self.corr = _small_cpu_corr()

    def test_a_recycled_address_does_not_inherit_the_previous_digest(self):
        fingerprints, addresses = [], []
        for value in (1.0, 2.0):
            weights = np.full(64, value)
            weights.flags.writeable = False
            addresses.append(weights.__array_interface__["data"][0])
            fingerprints.append(self.corr._fingerprint_weights(weights))
            del weights
            gc.collect()
        if addresses[0] != addresses[1]:
            self.skipTest("the allocator did not reuse the address")
        self.assertNotEqual(fingerprints[0], fingerprints[1])

    def test_the_same_frozen_array_still_hits_the_memo(self):
        from CosmoFuse import correlations as correlations_module

        weights = np.arange(64, dtype=np.float64)
        weights.flags.writeable = False
        first = self.corr._fingerprint_weights(weights)
        before = dict(correlations_module._FINGERPRINT_MEMO)
        second = self.corr._fingerprint_weights(weights)
        self.assertEqual(first, second)
        # a hit, not a rehash: the memo did not grow
        self.assertEqual(
            len(correlations_module._FINGERPRINT_MEMO), len(before)
        )

    def test_the_memo_pins_nothing(self):
        from CosmoFuse import correlations as correlations_module

        weights = np.arange(64, dtype=np.float64)
        weights.flags.writeable = False
        self.corr._fingerprint_weights(weights)
        key = next(
            k for k in correlations_module._FINGERPRINT_MEMO if k[0] == id(weights)
        )
        reference = correlations_module._FINGERPRINT_MEMO[key][0]
        del weights
        gc.collect()
        self.assertIsNone(reference(), "the memo kept the weight map alive")


class TestDeviceWeightFingerprint(unittest.TestCase):
    """The device branch used to return the CUDA pool pointer as identity.

    cupy's pool hands a freed pointer straight to the next allocation, so a
    plain ``for k: w = cp.asarray(w_host[k])`` loop gave the second map the
    first one's address -- and its cached sum of weights.  Emulated here so
    the gate runs without a GPU: two stand-in arrays reporting the *same*
    pointer must still fingerprint differently.
    """

    class _FakeDevice:
        id = 0

    class _FakeDeviceArray(np.ndarray):
        """A numpy array that looks native to a cupy backend."""

        device = None
        data = None

    def _fake(self, values, pointer):
        array = np.asarray(values, dtype=np.float64).view(self._FakeDeviceArray)
        array.device = self._FakeDevice()
        array.data = type("_Ptr", (), {"ptr": pointer})()
        return array

    def setUp(self):
        self.corr = _small_cpu_corr()
        self.corr.backend.name = "cupy"
        self.corr.backend.module = type(
            "_FakeModule", (), {"ndarray": self._FakeDeviceArray}
        )
        self.corr.backend.device_id = 0

    def test_two_arrays_at_one_pointer_fingerprint_differently(self):
        first = self._fake([1.0, 2.0], 0xDEADBEEF)
        second = self._fake([3.0, 4.0], 0xDEADBEEF)
        self.assertNotEqual(
            self.corr._fingerprint_weights(first),
            self.corr._fingerprint_weights(second),
        )

    def test_the_same_array_keeps_one_fingerprint(self):
        array = self._fake([1.0, 2.0], 0xDEADBEEF)
        self.assertEqual(
            self.corr._fingerprint_weights(array),
            self.corr._fingerprint_weights(array),
        )


class TestCombinationLayoutCache(unittest.TestCase):
    """``_combination_layout`` caches by array identity.

    The canonical combination arrays are cached by the orchestrator, but an
    explicit ``ggl_bin_combinations`` selection builds fresh ones on every
    call and drops them.  The layout carries ``rows`` -- the canonical row of
    every requested combination -- so a stale hit scatters the results into
    the previous selection's rows.
    """

    def setUp(self):
        from CosmoFuse.backend import _COMBINATION_LAYOUT_CACHE

        _COMBINATION_LAYOUT_CACHE.clear()

    @staticmethod
    def _layout(selection, n1, n2):
        from CosmoFuse.backend import _combination_layout

        comb_i = np.asarray([a for a, _ in selection], dtype=np.int32)
        comb_j = np.asarray([b for _, b in selection], dtype=np.int32)
        layout = _combination_layout(comb_i, comb_j, "cartesian", n1, n2)
        return layout, (id(comb_i), id(comb_j))

    def test_two_selections_do_not_share_a_layout(self):
        first = [(0, 0), (1, 1), (2, 2)]
        second = [(0, 2), (1, 0), (2, 1)]

        truths = {}
        for name, selection in (("first", first), ("second", second)):
            from CosmoFuse.backend import _COMBINATION_LAYOUT_CACHE

            _COMBINATION_LAYOUT_CACHE.clear()
            layout, _ = self._layout(selection, 3, 3)
            truths[name] = np.asarray(layout.rows).copy()
        self.assertFalse(np.array_equal(truths["first"], truths["second"]))

        from CosmoFuse.backend import _COMBINATION_LAYOUT_CACHE

        _COMBINATION_LAYOUT_CACHE.clear()
        for i in range(400):
            name, selection = (
                ("first", first) if i % 2 == 0 else ("second", second)
            )
            layout, _ = self._layout(selection, 3, 3)
            np.testing.assert_array_equal(
                np.asarray(layout.rows), truths[name],
                err_msg=f"iteration {i} got the other selection's layout",
            )

    def test_the_same_arrays_still_hit(self):
        from CosmoFuse.backend import _COMBINATION_LAYOUT_CACHE, _combination_layout

        comb_i = np.asarray([0, 1, 2], dtype=np.int32)
        comb_j = np.asarray([2, 0, 1], dtype=np.int32)
        first = _combination_layout(comb_i, comb_j, "cartesian", 3, 3)
        size = len(_COMBINATION_LAYOUT_CACHE)
        second = _combination_layout(comb_i, comb_j, "cartesian", 3, 3)
        self.assertIs(first, second)
        self.assertEqual(len(_COMBINATION_LAYOUT_CACHE), size)
