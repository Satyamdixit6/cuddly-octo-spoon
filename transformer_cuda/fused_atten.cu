// attention_kernels.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <cmath>       
#include <limits>      

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                     \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// --- Forward Kernels ---

// Kernel to compute Q @ K^T (Batched Matrix Multiply)
// Output: attn_scores [B, H, L, L]
// Inputs: Q [B, H, L, Dk], K [B, H, L, Dk]
__global__ void qk_matmul_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ attn_scores,
    int B, int H, int L, int Dk)
{
    int batch_idx = blockIdx.z / H;
    int head_idx = blockIdx.z % H;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y; // Q sequence length
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x; // K sequence length

    if (batch_idx < B && head_idx < H && row_idx < L && col_idx < L) {
        float sum = 0.0f;
        const float* q_ptr = q + batch_idx * H * L * Dk + head_idx * L * Dk + row_idx * Dk;
        const float* k_ptr = k + batch_idx * H * L * Dk + head_idx * L * Dk + col_idx * Dk;

        // Dot product
        for (int i = 0; i < Dk; ++i) {
            sum += q_ptr[i] * k_ptr[i];
        }

        int score_idx = batch_idx * H * L * L + head_idx * L * L + row_idx * L + col_idx;
        attn_scores[score_idx] = sum;
    }
}

// Kernel for Scaling, Masking, and Softmax (row-wise on attn_scores)
// In-place: attn_scores [B, H, L, L] -> attn_probs
// Input: mask [B, H, L, L] (optional, 0 for masked, 1 for unmasked)
__global__ void scale_mask_softmax_kernel(
    float* __restrict__ attn_scores,
    const float* __restrict__ mask,
    float scale,
    int B, int H, int L)
{
    int batch_idx = blockIdx.y / H;
    int head_idx = blockIdx.y % H;
    int row_idx = blockIdx.x;

    if (batch_idx < B && head_idx < H && row_idx < L) {
        int base_idx = batch_idx * H * L * L + head_idx * L * L + row_idx * L;
        float* row_scores = attn_scores + base_idx;
        const float* row_mask = mask ? (mask + base_idx) : nullptr;

        // Shared memory for max and sum reductions
        extern __shared__ float shared_mem[];
        float* shared_max = shared_mem;                // First block for max
        float* shared_sum = shared_mem + blockDim.x;   // Second block for sum

        // --- Find Max ---
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = threadIdx.x; j < L; j += blockDim.x) {
            float score = row_scores[j] * scale;
            if (row_mask && row_mask[j] == 0.0f) {
                score = -std::numeric_limits<float>::infinity();
            }
            row_scores[j] = score; // Store scaled score
            max_val = fmaxf(max_val, score);
        }
        shared_max[threadIdx.x] = max_val;
        __syncthreads();

        // Reduce max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        max_val = shared_max[0];

        // --- Exponentiate and Sum ---
        float sum_exp = 0.0f;
        shared_sum[threadIdx.x] = 0.0f;
        for (int j = threadIdx.x; j < L; j += blockDim.x) {
            float val = row_scores[j] > -std::numeric_limits<float>::infinity() ?
                        expf(row_scores[j] - max_val) : 0.0f;
            row_scores[j] = val;
            shared_sum[threadIdx.x] += val;
        }
        __syncthreads();

        // Reduce sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }
        sum_exp = shared_sum[0];

        // --- Normalize ---
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int j = threadIdx.x; j < L; j += blockDim.x) {
            row_scores[j] *= inv_sum; // Now contains probabilities
        }
    }
}

// Kernel to compute O = P @ V (Batched Matrix Multiply)
// Output: attn_output [B, H, L, Dv]
// Inputs: attn_probs (P) [B, H, L, L], V [B, H, L, Dv]
__global__ void v_matmul_kernel(
    const float* __restrict__ attn_probs,
    const float* __restrict__ v,
    float* __restrict__ attn_output,
    int B, int H, int L, int Dv)
{
    int batch_idx = blockIdx.z / H;
    int head_idx = blockIdx.z % H;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y; // Sequence length
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x; // Value dimension

    if (batch_idx < B && head_idx < H && row_idx < L && col_idx < Dv) {
        float sum = 0.0f;
        const float* prob_ptr = attn_probs + batch_idx * H * L * L + head_idx * L * L + row_idx * L;
        const float* v_ptr = v + batch_idx * H * L * Dv + head_idx * L * Dv;

        // Dot product: sum over sequence length
        for (int k = 0; k < L; ++k) {
            sum += prob_ptr[k] * v_ptr[k * Dv + col_idx];
        }

        int out_idx = batch_idx * H * L * Dv + head_idx * L * Dv + row_idx * Dv + col_idx;
        attn_output[out_idx] = sum;
    }
}

// --- Backward Kernels ---

// Kernel to compute dV = P^T @ dOut
__global__ void dv_kernel(
    const float* __restrict__ attn_probs,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_v,
    int B, int H, int L, int Dv)
{
    int batch_idx = blockIdx.z / H;
    int head_idx = blockIdx.z % H;
    int row_idx_k = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx_dv = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < B && head_idx < H && row_idx_k < L && col_idx_dv < Dv) {
        float sum = 0.0f;
        const float* prob_ptr_base = attn_probs + batch_idx * H * L * L + head_idx * L * L;
        const float* grad_out_base = grad_out + batch_idx * H * L * Dv + head_idx * L * Dv;

        for (int i = 0; i < L; ++i) {
            float p_ik = prob_ptr_base[i * L + row_idx_k];
            float dout_idv = grad_out_base[i * Dv + col_idx_dv];
            sum += p_ik * dout_idv;
        }

        int grad_v_idx = batch_idx * H * L * Dv + head_idx * L * Dv + row_idx_k * Dv + col_idx_dv;
        grad_v[grad_v_idx] = sum;
    }
}

// Kernel to compute dP = dOut @ V^T
__global__ void dp_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ v,
    float* __restrict__ grad_attn_probs,
    int B, int H, int L, int Dv)
{
    int batch_idx = blockIdx.z / H;
    int head_idx = blockIdx.z % H;
    int row_idx_i = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx_j = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < B && head_idx < H && row_idx_i < L && col_idx_j < L) {
        float sum = 0.0f;
        const float* grad_out_base = grad_out + batch_idx * H * L * Dv + head_idx * L * Dv;
        const float* v_base = v + batch_idx * H * L * Dv + head_idx * L * Dv;

        for (int k_dv = 0; k_dv < Dv; ++k_dv) {
            float dout_ik = grad_out_base[row_idx_i * Dv + k_dv];
            float v_jk = v_base[col_idx_j * Dv + k_dv];
            sum += dout_ik * v_jk;
        }

        int dp_idx = batch_idx * H * L * L + head_idx * L * L + row_idx_i * L + col_idx_j;
        grad_attn_probs[dp_idx] = sum;
    }
}

// Kernel to compute dS = (dP - sum(dP * P)) * P
__global__ void softmax_backward_kernel(
    float* __restrict__ grad_attn_probs,
    const float* __restrict__ attn_probs,
    int B, int H, int L)
{
    int batch_idx = blockIdx.y / H;
    int head_idx = blockIdx.y % H;
    int row_idx = blockIdx.x;

    if (batch_idx < B && head_idx < H && row_idx < L) {
        int base_idx = batch_idx * H * L * L + head_idx * L * L + row_idx * L;
        float* row_grad_p = grad_attn_probs + base_idx;
        const float* row_p = attn_probs + base_idx;

        // Shared memory for reduction
        extern __shared__ float shared_sum[];
        shared_sum[threadIdx.x] = 0.0f;

        for (int j = threadIdx.x; j < L; j += blockDim.x) {
            shared_sum[threadIdx.x] += row_grad_p[j] * row_p[j];
        }
        __syncthreads();

        // Reduce sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float row_sum = shared_sum[0];

        // Compute dS
        for (int j = threadIdx.x; j < L; j += blockDim.x) {
            row_grad_p[j] = (row_grad_p[j] - row_sum) * row_p[j];
        }
    }
}

// Kernel to compute dQ = dS @ K
__global__ void dq_kernel(
    const float* __restrict__ grad_scores,
    const float* __restrict__ k,
    float* __restrict__ grad_q,
    float scale,
    int B, int H, int L, int Dk)
{
    int batch_idx = blockIdx.z / H;
    int head_idx = blockIdx.z % H;
    int row_idx_i = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx_dk = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < B && head_idx < H && row_idx_i < L && col_idx_dk < Dk) {
        float sum = 0.0f;
        const float* grad_scores_base = grad_scores + batch_idx * H * L * L + head_idx * L * L;
        const float* k_base = k + batch_idx * H * L * Dk + head_idx * L * Dk;

        for (int j = 0; j < L; ++j) {
            float ds_ij = grad_scores_base[row_idx_i * L + j];
            float k_jdk = k_base[j * Dk + col_idx_dk];
            sum += ds_ij * k_jdk;
        }

        int grad_q_idx = batch_idx * H * L * Dk + head_idx * L * Dk + row_idx_i * Dk + col_idx_dk;
        grad_q[grad_q_idx] = sum * scale;
    }
}

// Kernel to compute dK = dS^T @ Q
__global__ void dk_kernel(
    const float* __restrict__ grad_scores,
    const float* __restrict__ q,
    float* __restrict__ grad_k,
    float scale,
    int B, int H, int L, int Dk)
{
    int batch_idx = blockIdx.z / H;
    int head_idx = blockIdx.z % H;
    int row_idx_j = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx_dk = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < B && head_idx < H && row_idx_j < L && col_idx_dk < Dk) {
        float sum = 0.0f;
        const float* grad_scores_base = grad_scores + batch_idx * H * L * L + head_idx * L * L;
        const float* q_base = q + batch_idx * H * L * Dk + head_idx * L * Dk;

        for (int i = 0; i < L; ++i) {
            float ds_ij = grad_scores_base[i * L + row_idx_j];
            float q_idk = q_base[i * Dk + col_idx_dk];
            sum += ds_ij * q_idk;
        }

        int grad_k_idx = batch_idx * H * L * Dk + head_idx * L * Dk + row_idx_j * Dk + col_idx_dk;
        grad_k[grad_k_idx] = sum * scale;
    }
}