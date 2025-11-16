#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

#define WARP_SIZE 32
#define MAX_THREADS 1024

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void mqa_forward_kernel(
    const float* Q,      // [B, N, H, D]
    const float* K,      // [B, S, D]
    const float* V,      // [B, S, D]
    float* O,            // [B, N, H, D]
    float* attn_scores,  // [B, H, N, S] - saved for backward
    int B, int N, int S, int H, int D
){

    int b = blockIdx.z;
    int h = blockIdx.y;
    int n = blockIdx.x;
    
    if (b >= B || h >= H || n >= N) return;

    extern __shared__ float smem[];
    float* s_K = smem;                    // [S, D]
    float* s_V = smem + S * D;            // [S, D]
    float* s_scores = smem + 2 * S * D;

    const float scale = rsqrtf((float)D);
    // Load KV
    for (int i = threadIdx.x; i < S * D; i += blockDim.x) {
        int s_idx = i / D;
        int d_idx = i % D;
        s_K[i] = K[b * S * D + s_idx * D + d_idx];
        s_V[i] = V[b * S * D + s_idx * D + d_idx];
    }
    __syncthreads();
     // Compute attention scores: Q[n,h,:] @ K[s,:].T
     for (int s = threadIdx.x; s < S; s += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < D; d++) {
            float q_val = Q[b * N * H * D + n * H * D + h * D + d];
            float k_val = s_K[s * D + d];
            score += q_val * k_val;
        }
        s_scores[s] = score * scale;
    }
    
    //softmax
    float max_score = -INFINITY;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        max_score = fmaxf(max_score, s_scores[s]);
    }

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -INFINITY;
    __syncthreads();
    
    atomicMax((int*)&s_max, __float_as_int(max_score));
    __syncthreads();
    

    float sum_exp = 0.0f;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        s_scores[s] = expf(s_scores[s] - s_max);
        sum_exp += s_scores[s];
    }
}