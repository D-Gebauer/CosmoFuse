#define BLOCK_SIZE 256

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
