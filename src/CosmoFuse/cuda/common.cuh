/*
 * common.cuh -- Shared CUDA utilities for CosmoFuse correlation kernels.
 *
 * Provides thread-block level parallel reductions used to sum per-pair
 * contributions (e.g. weighted shear products, density products) across
 * all pixel pairs within an angular separation bin.
 */

#define BLOCK_SIZE 256

/*
 * Parallel sum reduction within a single thread block.
 * Each thread contributes its local accumulator `val` (e.g. a partial
 * ξ+ numerator); the result in thread 0 is the total sum for the block.
 */
template<typename T>
__device__ inline T block_reduce_sum(T val) {
    __shared__ T shared[BLOCK_SIZE];
    int lane = threadIdx.x;
    shared[lane] = val;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

/*
 * Simultaneous reduction of two values in one pass -- avoids a second
 * __syncthreads() barrier.  Used when a kernel accumulates two quantities
 * over the same pair loop, e.g. ξ+ and ξ- numerators, or a numerator
 * and its weight denominator.
 */
template<typename T>
__device__ inline void block_reduce_sum_pair(T val1, T val2, T* out1, T* out2) {
    __shared__ T s1[BLOCK_SIZE];
    __shared__ T s2[BLOCK_SIZE];
    int lane = threadIdx.x;
    s1[lane] = val1;
    s2[lane] = val2;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            s1[lane] += s1[lane + stride];
            s2[lane] += s2[lane + stride];
        }
        __syncthreads();
    }
    *out1 = s1[0];
    *out2 = s2[0];
}

/*
 * Simultaneous reduction of three values in one pass -- avoids two extra
 * __syncthreads() barrier chains.  Used when a kernel accumulates three
 * independent quantities over the same pair loop, e.g. ξ+ numerator,
 * ξ- numerator, and the shared weight denominator.
 */
template<typename T>
__device__ inline void block_reduce_sum_triple(T val1, T val2, T val3,
                                               T* out1, T* out2, T* out3) {
    __shared__ T s1[BLOCK_SIZE];
    __shared__ T s2[BLOCK_SIZE];
    __shared__ T s3[BLOCK_SIZE];
    int lane = threadIdx.x;
    s1[lane] = val1;
    s2[lane] = val2;
    s3[lane] = val3;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            s1[lane] += s1[lane + stride];
            s2[lane] += s2[lane + stride];
            s3[lane] += s3[lane + stride];
        }
        __syncthreads();
    }
    *out1 = s1[0];
    *out2 = s2[0];
    *out3 = s3[0];
}
