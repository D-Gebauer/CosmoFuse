__COMMON_CUDA_SOURCE__

#include <cuComplex.h>

extern "C" __global__
void __KERNEL_NAME__(
    const __MAP_C_TYPE__* density,
    const __MAP_C_TYPE__* shear,
    const __MAP_C_TYPE__* density_w,
    const __MAP_C_TYPE__* shear_w,
    const long long* ind_i,
    const long long* ind_j,
    const __COMPLEX_C_TYPE__* rot_i,
    const __COMPLEX_C_TYPE__* rot_j,
    const long long* pair_offsets,
    const long long nbins_total,
    const int n_density_bins,
    const int n_shear_bins,
    const int npatches,
    const int npix,
    const unsigned int* q_inds,
    const __MAP_C_TYPE__* q_cos,
    const __MAP_C_TYPE__* q_sin,
    const __MAP_C_TYPE__* q_val,
    const long long* q_offsets,
    const __MAP_C_TYPE__* q_patch_area,
    const int* ss_comb_i,
    const int* ss_comb_j,
    const int n_ss_comb,
    const int* dd_comb_i,
    const int* dd_comb_j,
    const int n_dd_comb,
    const int* ds_comb_i,
    const int* ds_comb_j,
    const int n_ds_comb,
    __MAP_C_TYPE__* out_ma_num,
    __MAP_C_TYPE__* out_ma_den,
    __MAP_C_TYPE__* out_mg_num,
    __MAP_C_TYPE__* out_mg_den,
    __MAP_C_TYPE__* out_xip_num,
    __MAP_C_TYPE__* out_xim_num,
    __MAP_C_TYPE__* out_xipm_den,
    __MAP_C_TYPE__* out_xig_num,
    __MAP_C_TYPE__* out_xig_den,
    __MAP_C_TYPE__* out_xit_num,
    __MAP_C_TYPE__* out_xit_den)
{
    const int lane = (int)threadIdx.x;
    const long long x = (long long)blockIdx.x;
    const int y = (int)blockIdx.y;
    const int z = (int)blockIdx.z;

    if (z == 0) {
        if (x >= npatches || y >= n_shear_bins) return;
        const long long start = q_offsets[x];
        const long long stop = q_offsets[x + 1];
        __MAP_C_TYPE__ sum_num = (__MAP_C_TYPE__)0.0;
        __MAP_C_TYPE__ sum_den = (__MAP_C_TYPE__)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const unsigned int pix = q_inds[idx];
            const long long shear_idx = ((long long)pix * (long long)n_shear_bins + (long long)y) * 2LL;
            const long long w_idx = (long long)pix * (long long)n_shear_bins + (long long)y;
            const __MAP_C_TYPE__ g1 = shear[shear_idx];
            const __MAP_C_TYPE__ g2 = shear[shear_idx + 1LL];
            const __MAP_C_TYPE__ wv = shear_w[w_idx];
            const __MAP_C_TYPE__ gt = -g1 * q_cos[idx] - g2 * q_sin[idx];
            sum_num += wv * gt * q_val[idx];
            sum_den += wv;
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * (long long)npatches + x;
            out_ma_num[out_idx] = q_patch_area[x] * sum_num;
            out_ma_den[out_idx] = sum_den;
        }
        return;
    }

    if (z == 1) {
        if (x >= npatches || y >= n_density_bins) return;
        const long long start = q_offsets[x];
        const long long stop = q_offsets[x + 1];
        __MAP_C_TYPE__ sum_num = (__MAP_C_TYPE__)0.0;
        __MAP_C_TYPE__ sum_den = (__MAP_C_TYPE__)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const unsigned int pix = q_inds[idx];
            const long long d_idx = (long long)pix * (long long)n_density_bins + (long long)y;
            const __MAP_C_TYPE__ wv = density_w[d_idx];
            sum_num += wv * density[d_idx] * q_val[idx];
            sum_den += wv;
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * (long long)npatches + x;
            out_mg_num[out_idx] = q_patch_area[x] * sum_num;
            out_mg_den[out_idx] = sum_den;
        }
        return;
    }

    if (z == 2) {
        if (x >= nbins_total || y >= (2 * n_ss_comb)) return;
        const int comb_idx = y >> 1;
        const int ori = y & 1;
        const int i = ss_comb_i[comb_idx];
        const int j = ss_comb_j[comb_idx];
        if (ori == 1 && i == j) return;

        int ai = i;
        int bj = j;
        if (ori == 1 && i != j) {
            ai = j;
            bj = i;
        }

        const long long start = pair_offsets[x];
        const long long stop = pair_offsets[x + 1];
        __MAP_C_TYPE__ sum_p = (__MAP_C_TYPE__)0.0;
        __MAP_C_TYPE__ sum_m = (__MAP_C_TYPE__)0.0;
        __MAP_C_TYPE__ sum_w = (__MAP_C_TYPE__)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const long long pix_a = ind_i[idx];
            const long long pix_b = ind_j[idx];
            const __COMPLEX_C_TYPE__ ex_a = rot_i[idx];
            const __COMPLEX_C_TYPE__ ex_b = rot_j[idx];

            const long long a_base = ((pix_a * (long long)n_shear_bins + (long long)ai) * 2LL);
            const long long b_base = ((pix_b * (long long)n_shear_bins + (long long)bj) * 2LL);
            const __MAP_C_TYPE__ ga1 = shear[a_base];
            const __MAP_C_TYPE__ ga2 = shear[a_base + 1LL];
            const __MAP_C_TYPE__ gb1 = shear[b_base];
            const __MAP_C_TYPE__ gb2 = shear[b_base + 1LL];

            const __MAP_C_TYPE__ a_r = ga1 * ex_a.x - ga2 * ex_a.y;
            const __MAP_C_TYPE__ a_i = ga1 * ex_a.y + ga2 * ex_a.x;
            const __MAP_C_TYPE__ b_r = gb1 * ex_b.x - gb2 * ex_b.y;
            const __MAP_C_TYPE__ b_i = gb1 * ex_b.y + gb2 * ex_b.x;

            const long long wa_idx = pix_a * (long long)n_shear_bins + (long long)ai;
            const long long wb_idx = pix_b * (long long)n_shear_bins + (long long)bj;
            const __MAP_C_TYPE__ wv = shear_w[wa_idx] * shear_w[wb_idx];
            sum_w += wv;
            sum_p += wv * (b_r * a_r + b_i * a_i);
            sum_m += wv * (b_r * a_r - b_i * a_i);
        }
        block_reduce_sum_pair(sum_p, sum_m, &sum_p, &sum_m);
        sum_w = block_reduce_sum(sum_w);
        if (lane == 0) {
            const long long out_idx = (long long)y * nbins_total + x;
            out_xip_num[out_idx] = sum_p;
            out_xim_num[out_idx] = sum_m;
            out_xipm_den[out_idx] = sum_w;
        }
        return;
    }

    if (z == 3) {
        if (x >= nbins_total || y >= (2 * n_dd_comb)) return;
        const int comb_idx = y >> 1;
        const int ori = y & 1;
        const int i = dd_comb_i[comb_idx];
        const int j = dd_comb_j[comb_idx];
        if (ori == 1 && i == j) return;

        int ai = i;
        int bj = j;
        if (ori == 1 && i != j) {
            ai = j;
            bj = i;
        }

        const long long start = pair_offsets[x];
        const long long stop = pair_offsets[x + 1];
        __MAP_C_TYPE__ sum_num = (__MAP_C_TYPE__)0.0;
        __MAP_C_TYPE__ sum_den = (__MAP_C_TYPE__)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const long long pix_a = ind_i[idx];
            const long long pix_b = ind_j[idx];
            const long long ia = pix_a * (long long)n_density_bins + (long long)ai;
            const long long jb = pix_b * (long long)n_density_bins + (long long)bj;
            const __MAP_C_TYPE__ wv = density_w[ia] * density_w[jb];
            sum_den += wv;
            sum_num += wv * density[ia] * density[jb];
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * nbins_total + x;
            out_xig_num[out_idx] = sum_num;
            out_xig_den[out_idx] = sum_den;
        }
        return;
    }

    if (z == 4) {
        if (x >= nbins_total || y >= n_ds_comb) return;
        const int lens_bin = ds_comb_i[y];
        const int source_bin = ds_comb_j[y];
        const long long start = pair_offsets[x];
        const long long stop = pair_offsets[x + 1];
        __MAP_C_TYPE__ sum_num = (__MAP_C_TYPE__)0.0;
        __MAP_C_TYPE__ sum_den = (__MAP_C_TYPE__)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const long long pix_a = ind_i[idx];
            const long long pix_b = ind_j[idx];

            const long long lens_ab = pix_a * (long long)n_density_bins + (long long)lens_bin;
            const long long src_ab = pix_b * (long long)n_shear_bins + (long long)source_bin;
            const long long src_ab_base = src_ab * 2LL;
            const __COMPLEX_C_TYPE__ ex_ab = rot_j[idx];
            const __MAP_C_TYPE__ gt_ab = -shear[src_ab_base] * ex_ab.x + shear[src_ab_base + 1LL] * ex_ab.y;
            const __MAP_C_TYPE__ w_ab = density_w[lens_ab] * shear_w[src_ab];
            sum_num += w_ab * density[lens_ab] * gt_ab;
            sum_den += w_ab;

            const long long lens_ba = pix_b * (long long)n_density_bins + (long long)lens_bin;
            const long long src_ba = pix_a * (long long)n_shear_bins + (long long)source_bin;
            const long long src_ba_base = src_ba * 2LL;
            const __COMPLEX_C_TYPE__ ex_ba = rot_i[idx];
            const __MAP_C_TYPE__ gt_ba = -shear[src_ba_base] * ex_ba.x + shear[src_ba_base + 1LL] * ex_ba.y;
            const __MAP_C_TYPE__ w_ba = density_w[lens_ba] * shear_w[src_ba];
            sum_num += w_ba * density[lens_ba] * gt_ba;
            sum_den += w_ba;
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * nbins_total + x;
            out_xit_num[out_idx] = sum_num;
            out_xit_den[out_idx] = sum_den;
        }
        return;
    }
}
