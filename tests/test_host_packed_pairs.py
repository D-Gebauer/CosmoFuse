"""Host-side payload packing (``pack_host_pairs=True``, idea #11).

The host arrays and the pair file then hold 8 instead of 24 bytes per pair.
What has to hold:

* the device state is *bitwise* the one built by packing inside ``prepare()``
  -- packing on global ids and packing on device rows must agree, otherwise
  the two options would measure different estimators;
* the unpacked CPU arrays rebuilt from the packed payload are the ones the
  CPU backend uses with ``pack_pairs=True``;
* a packed file round-trips, slices by patch (treecode renumbering included)
  and carries its own format version;
* nothing changes by default.
"""

import os
import tempfile
import unittest

import h5py
import healpy as hp
import numpy as np

from CosmoFuse import Correlation

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)


def make_mask():
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (np.degrees(phi) < 200.0) & (np.abs(90 - np.degrees(theta)) < 55)
    rng = np.random.default_rng(1)
    mask[rng.choice(NPIX, NPIX // 30, replace=False)] = False
    return mask


MASK = make_mask()
KWARGS = dict(
    patch_size=900,
    theta_Q=300,
    f_mask=0.3,
    nbins=4,
    theta_min=200,
    theta_max=1600,
    device="cpu",
    map_precision="float64",
    rotation_precision="float64",
    resolution_factor=2.0,
)


def make_setup(preprocess=True, **kwargs):
    corr = Correlation.from_mask(NSIDE, MASK, 2, **{**KWARGS, **kwargs})
    if preprocess:
        corr.preprocess()
    return corr


def device_state(corr):
    ctx = corr.compute_context
    return dict(
        packed_pairs=ctx.packed_pairs_dev,
        packed_row_base=ctx.packed_row_base_dev,
        packed_perm=ctx.packed_perm_dev,
        inds=corr.inds_dev,
        exp2phi=corr.exp2phi_dev,
        bins=corr.bins_dev,
        tot_bins=corr.tot_bins_reduceat_dev,
    )


def assert_same_state(case, a, b, keys=None):
    for key in keys or a:
        x, y = a[key], b[key]
        if x is None or y is None:
            case.assertIs(x, y, key)
            continue
        x, y = np.asarray(x), np.asarray(y)
        case.assertEqual(x.dtype, y.dtype, key)
        case.assertTrue(np.array_equal(x, y), key)


class TestHostPacking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_packed = make_setup(pack_pairs=True)
        cls.host_packed = make_setup(pack_pairs=True, pack_host_pairs=True)
        cls.host_only = make_setup(pack_pairs=False, pack_host_pairs=True)
        cls.exact = make_setup()

    def test_off_by_default(self):
        corr = self.exact
        self.assertFalse(corr.pack_host_pairs)
        self.assertIsNone(corr.packed_pairs)
        self.assertIsNotNone(corr.pair_inds)
        self.assertEqual(len(corr.pair_inds), corr.n_patches)

    def test_host_arrays_are_replaced_by_the_packed_payload(self):
        corr = self.host_packed
        self.assertIsNone(corr.pair_inds)
        self.assertIsNone(corr.pair_exp2phi)
        self.assertEqual(len(corr.packed_pairs), corr.n_patches)
        payload = sum(p.nbytes for p in corr.packed_pairs)
        self.assertEqual(payload, 8 * corr.ntotpairs)
        # ... against 2 indices + 2 rotation factors per pair (24 B/pair at
        # the default float32 rotations, 40 B here at float64)
        exact = self.exact
        exact_bytes = sum(
            a.nbytes + b.nbytes
            for a, b in zip(exact.pair_inds, exact.pair_exp2phi)
        )
        per_pair = 2 * exact.index_dtype.itemsize + 2 * np.dtype(
            exact.rotation_complex_dtype
        ).itemsize
        self.assertEqual(exact_bytes, per_pair * exact.ntotpairs)
        self.assertLessEqual(payload * 3, exact_bytes)
        # the row blocks are a rounding error on top
        blocks = sum(b.nbytes for b in corr.packed_block_ids)
        self.assertLess(blocks, 0.05 * payload)

    def test_device_state_matches_packing_inside_prepare(self):
        self.assertGreater(self.device_packed.ntotpairs, 1000)
        assert_same_state(
            self, device_state(self.device_packed), device_state(self.host_packed)
        )

    def test_unpacked_cpu_arrays_match(self):
        """pack_pairs=False keeps the 24 B device arrays; rebuilt from the
        packed payload they are the ones pack_pairs=True uses on CPU."""
        self.assertIsNone(self.host_only.compute_context.packed_pairs_dev)
        assert_same_state(
            self,
            device_state(self.device_packed),
            device_state(self.host_only),
            keys=("inds", "exp2phi", "bins", "tot_bins"),
        )

    def test_measurements_match_prepare_time_packing(self):
        rng = np.random.default_rng(4)
        n = self.exact.n_active
        shear = rng.normal(size=(2, 2, n)) * 0.3
        w = rng.uniform(0.2, 2.0, size=(2, n))
        ref = self.device_packed.get_full_tomo_shear(shear, w, return_device=False)
        for corr in (self.host_packed, self.host_only):
            got = corr.get_full_tomo_shear(shear, w, return_device=False)
            for x, y in zip(ref, got):
                self.assertTrue(np.array_equal(np.asarray(x), np.asarray(y)))

    def test_release_host_pairs(self):
        corr = make_setup(preprocess=False, pack_pairs=True, pack_host_pairs=True)
        corr.preprocess(release_host_pairs=True)
        self.assertIsNone(corr.packed_pairs)
        self.assertIsNone(corr.packed_block_ids)
        assert_same_state(self, device_state(self.device_packed), device_state(corr))

    def test_pickle_round_trip(self):
        import pickle

        clone = pickle.loads(pickle.dumps(self.host_packed))
        self.assertTrue(clone.pack_host_pairs)
        clone.prepare()
        assert_same_state(self, device_state(self.host_packed), device_state(clone))


class TestPackedPairFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.reference = make_setup(pack_pairs=True)
        cls.source = make_setup(pack_pairs=True, pack_host_pairs=True)
        cls.path = os.path.join(cls.tmp.name, "packed.h5")
        cls.source.save_pairs(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _fresh(self, **kwargs):
        return make_setup(preprocess=False, **kwargs)

    def test_file_marks_itself_packed(self):
        with h5py.File(self.path, "r") as fp:
            self.assertEqual(int(fp.attrs["format_version"]), 4)
            self.assertTrue(bool(fp.attrs["packed_pairs"]))
            self.assertIn("packed_pairs", fp)
            self.assertIn("packed_block_ids", fp)
            self.assertNotIn("pair_exp2phi", fp)
            self.assertNotIn("tc_pair_inds", fp)
            self.assertEqual(fp["packed_pairs"].dtype, np.uint16)
            # smaller than the exact file it replaces
            exact_path = os.path.join(self.tmp.name, "exact.h5")
        make_setup().save_pairs(exact_path)
        self.assertLess(
            os.path.getsize(self.path), 0.6 * os.path.getsize(exact_path)
        )

    def test_round_trip(self):
        corr = self._fresh(pack_pairs=True)
        corr.load_pairs(self.path)
        self.assertTrue(corr.pack_host_pairs)
        self.assertIsNone(corr.pair_inds)
        self.assertEqual(corr.ntotpairs, self.reference.ntotpairs)
        assert_same_state(self, device_state(self.reference), device_state(corr))

    def test_round_trip_without_device_packing(self):
        corr = self._fresh(pack_pairs=False)
        corr.load_pairs(self.path)
        assert_same_state(
            self,
            device_state(self.reference),
            device_state(corr),
            keys=("inds", "exp2phi", "bins", "tot_bins"),
        )

    def test_sliced_load_matches_the_same_patches(self):
        n = self.source.n_patches
        self.assertGreater(n, 3)
        start, stop = 1, n - 1
        part = self._fresh(pack_pairs=True)
        part.load_pairs(self.path, start_ind=start, stop_ind=stop)
        self.assertEqual(part.n_patches, stop - start)

        whole = self._fresh(pack_pairs=True)
        whole.load_pairs(self.path)
        offsets = np.asarray(whole.tot_bins_reduceat_dev)
        p0 = int(offsets[start * whole.nbins])
        p1 = int(offsets[stop * whole.nbins])
        self.assertEqual(part.ntotpairs, p1 - p0)
        np.testing.assert_array_equal(
            np.asarray(part.compute_context.packed_pairs_dev),
            np.asarray(whole.compute_context.packed_pairs_dev)[p0:p1],
        )
        # the coarse cells are renumbered by the slice; what must not change
        # is what the rows point at, i.e. the measurement of those patches
        rng = np.random.default_rng(7)
        shear = rng.normal(size=(2, 2, whole.n_active)) * 0.3
        w = rng.uniform(0.2, 2.0, size=(2, whole.n_active))
        ref = whole.get_full_tomo_shear(shear, w, return_device=False)
        got = part.get_full_tomo_shear(shear, w, return_device=False)
        np.testing.assert_array_equal(got[0], np.asarray(ref[0])[..., start:stop])
        for i in (1, 2):
            np.testing.assert_array_equal(
                got[i], np.asarray(ref[i])[..., start:stop, :]
            )

    def test_exact_file_clears_a_previously_packed_payload(self):
        exact_path = os.path.join(self.tmp.name, "exact_reload.h5")
        make_setup().save_pairs(exact_path)
        corr = self._fresh(pack_pairs=True, pack_host_pairs=True)
        corr.load_pairs(self.path)
        self.assertIsNotNone(corr.packed_pairs)
        corr.load_pairs(exact_path)
        self.assertIsNone(corr.packed_pairs)
        self.assertIsNotNone(corr.pair_inds)


if __name__ == "__main__":
    unittest.main()
