#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        TORCH_CHECK(false, "CUDA error"); \
    } \
} while(0)

#define TILE_T 32
#define TILE_D 32

__global__ void modulate_kernel(const float* __restrict__ x,
                               const float* __restrict__ shift,
                               const float* __restrict__ scale,
                               float* __restrict__ out,
                               int T, int D) {
    int n = blockIdx.z;
    int t_start = blockIdx.x * TILE_T;
    int d_start = blockIdx.y * TILE_D;
    int t_idx = threadIdx.x;
    int d_idx = threadIdx.y;
    int t = t_start + t_idx;
    int d = d_start + d_idx;

    __shared__ float s_shared[TILE_D];
    __shared__ float sh_shared[TILE_D];
    for (int i = d_idx; i < TILE_D; i += blockDim.y) {
        int d_global = d_start + i;
        if (d_global < D) {
            s_shared[i] = scale[n * D + d_global];
            sh_shared[i] = shift[n * D + d_global];
        }
    }
    __syncthreads();

    if (t < T && d < D) {
        int idx = n * T * D + t * D + d;
        float s_val = s_shared[d_idx];
        float sh_val = sh_shared[d_idx];
        out[idx] = x[idx] * (1.0f + s_val) + sh_val;
    }
}

__global__ void sincos_pos_embed_kernel(float* __restrict__ pos_embed,
                                       int embed_dim, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_positions = grid_size * grid_size;
    if (idx < num_positions) {
        int i = idx / grid_size;
        int j = idx % grid_size;
        int D = embed_dim;
        int half = D / 2;
        int quarter = half / 2;
        float* out_ptr = pos_embed + idx * D;
        for (int k = 0; k < quarter; k++) {
            float omega = expf(-logf(10000.0f) * ((float)k / (float)half));
            float val = i * omega;
            out_ptr[k] = sinf(val);
            out_ptr[k + quarter] = cosf(val);
        }
        for (int k = 0; k < quarter; k++) {
            float omega = expf(-logf(10000.0f) * ((float)k / (float)half));
            float val = j * omega;
            out_ptr[half + k] = sinf(val);
            out_ptr[half + k + quarter] = cosf(val);
        }
    }
}

torch::Tensor modulate_forward(torch::Tensor x, torch::Tensor shift, torch::Tensor scale) {
    CHECK_CUDA(x); CHECK_CUDA(shift); CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(shift); CHECK_CONTIGUOUS(scale);

    int N = x.size(0);
    int T = x.size(1);
    int D = x.size(2);
    auto out = torch::empty_like(x);
    dim3 blockDim(TILE_T, TILE_D, 1);
    dim3 gridDim((T + TILE_T - 1) / TILE_T, (D + TILE_D - 1) / TILE_D, N);
    modulate_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        shift.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        T, D
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return out;
}

torch::Tensor sincos_pos_embed_forward(int embed_dim, int grid_size) {
    int num_positions = grid_size * grid_size;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto pos_embed = torch::empty({num_positions, embed_dim}, options);
    int threadsPerBlock = 256;
    int blocks = (num_positions + threadsPerBlock - 1) / threadsPerBlock;
    sincos_pos_embed_kernel<<<blocks, threadsPerBlock>>>(
        pos_embed.data_ptr<float>(), embed_dim, grid_size
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return pos_embed;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("modulate_forward", &modulate_forward, "Modulate forward (CUDA)");
    m.def("sincos_pos_embed_forward", &sincos_pos_embed_forward, "Sincos Pos Embed forward (CUDA)");
}