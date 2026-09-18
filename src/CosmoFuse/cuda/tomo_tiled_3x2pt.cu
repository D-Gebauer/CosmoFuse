/*
 * tomo_tiled_3x2pt.cu -- xi+-, xi_g and xi_t of the fused 3x2pt path in one
 *                        walk over the pairs (pair_tiles.cuh: tile_multi).
 *
 * DO_SS / DO_DD / DO_DS select the statistics at compile time; the library
 * launches (1, 1, 1) while all accumulators fit the register budget and
 * falls back to the three standalone tiles beyond it.  Output layouts are
 * those of the standalone tiled kernels.
 *
 * Grid: blockIdx.x = angular bin, threadIdx.x strides the pairs.
 */

__COMMON_CUDA_SOURCE__
__PAIR_TILES_CUDA_SOURCE__


template<typename T, typename C, int N_DENSITY, int N_SHEAR, int DD_AUTO_ONLY,
         int DO_SS, int DO_DD, int DO_DS, typename I, typename ACC>
__global__ void gpu_tiled_tomo_reduce_3x2pt(
    const T* density,
    const T* shear,
    const T* density_w,
    const T* shear_w,
    const I* ind_i,
    const I* ind_j,
    const C* rot_i,
    const C* rot_j,
    const long long* bin_offsets,
    ACC* out_xipm_num,
    ACC* out_xipm_den,
    ACC* out_xig_num,
    ACC* out_xig_den,
    ACC* out_xit_num,
    ACC* out_xit_den,
    const long long nbins_total)
{
    const long long bin_flat = (long long)blockIdx.x;
    if (bin_flat >= nbins_total) {
        return;
    }
    const UnpackedPairs<T, C, I> pairs = {ind_i, ind_j, rot_i, rot_j};
    tile_multi<T, N_DENSITY, N_SHEAR, DD_AUTO_ONLY, DO_SS, DO_DD, DO_DS, ACC>(
        pairs, density, shear, density_w, shear_w,
        bin_offsets[bin_flat], bin_offsets[bin_flat + 1],
        out_xipm_num, out_xipm_den, out_xig_num, out_xig_den, out_xit_num, out_xit_den,
        nbins_total, bin_flat);
}


template<typename T, int N_DENSITY, int N_SHEAR, int DD_AUTO_ONLY,
         int DO_SS, int DO_DD, int DO_DS, typename ACC>
__global__ void gpu_tiled_packed_reduce_3x2pt(
    const T* density,
    const T* shear,
    const T* density_w,
    const T* shear_w,
    const unsigned short* pairs,
    const long long* bin_offsets,
    const long long* row_base,
    ACC* out_xipm_num,
    ACC* out_xipm_den,
    ACC* out_xig_num,
    ACC* out_xig_den,
    ACC* out_xit_num,
    ACC* out_xit_den,
    const long long nbins_total)
{
    const long long bin_flat = (long long)blockIdx.x;
    if (bin_flat >= nbins_total) {
        return;
    }
    const PackedPairs<T> packed = {
        reinterpret_cast<const ushort4*>(pairs), row_base[bin_flat]};
    tile_multi<T, N_DENSITY, N_SHEAR, DD_AUTO_ONLY, DO_SS, DO_DD, DO_DS, ACC>(
        packed, density, shear, density_w, shear_w,
        bin_offsets[bin_flat], bin_offsets[bin_flat + 1],
        out_xipm_num, out_xipm_den, out_xig_num, out_xig_den, out_xit_num, out_xit_den,
        nbins_total, bin_flat);
}
