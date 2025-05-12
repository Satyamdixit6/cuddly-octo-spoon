// cuda_adaln_modulation.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h> // For cublasSgemm
#include <math.h>       // For rsqrtf, expf (or use CUDA's __expf, __rsqrtf)

// ============================================================================
// Helper Device Functions
// ============================================================================

/**
 * @brief Computes the SiLU (Sigmoid Linear Unit) activation function.
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * This is a CUDA device function, meant to be called from GPU kernels.
 * __forceinline__ suggests the compiler aggressively inline this function.
 * Using __expf for potentially faster float exponential on GPU.
 */
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x)); // Using CUDA's __expf
}

// ============================================================================
// SiLU + Linear Operation
// ============================================================================

// Note: The following kernel `silu_linear_kernel` seems like an earlier version or
// a component for a different structure. The main forward pass `silu_linear_forward`
// uses its own inline `silu_kernel` and `add_bias_kernel`.
// This kernel is kept for completeness if it was intended for another purpose.
/**
 * @brief CUDA kernel to apply a bias and then SiLU activation element-wise.
 * Operation: c_out[i] = SiLU(c_in[i] + bias_for_element_channel)
 * This might be one part of a fused MLP layer if the matrix multiplication were separate.
 *
 * @param c_in Input tensor data. Flattened view.
 * @param c_out Output tensor data. Flattened view.
 * @param bias Bias tensor data, applied per channel.
 * @param N First dimension (e.g., batch size).
 * @param dim Second dimension (e.g., feature dimension). Total elements = N * dim.
 */
__global__ void silu_linear_kernel(
    const float* __restrict__ c_in,
    float* __restrict__ c_out,
    const float* __restrict__ bias,
    int N,
    int dim
) {
    // Global thread index for the flattened N*dim tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * dim) {
        // Add bias (bias[idx % dim] ensures correct bias for the channel)
        float val = c_in[idx] + bias[idx % dim];
        // Apply SiLU
        val = silu(val);
        c_out[idx] = val;
    }
}

/**
 * @brief Performs a SiLU activation followed by a Linear transformation.
 * Operation: out = SiLU(c) @ w + b
 * This is a common pattern in MLP (Multi-Layer Perceptron) blocks.
 *
 * @param c Input tensor of shape [B, d] (Batch size B, input dimension d).
 * @param w Weight tensor of shape [d, out_dim] for the linear layer.
 * @param b Bias tensor of shape [out_dim] for the linear layer.
 * @return torch::Tensor Output tensor of shape [B, out_dim].
 */
torch::Tensor silu_linear_forward(
    torch::Tensor c,     // Input tensor [B, d]
    torch::Tensor w,     // Weight matrix [d, out_dim]
    torch::Tensor b      // Bias vector [out_dim]
) {
    // Input validation
    TORCH_CHECK(c.is_cuda() && w.is_cuda() && b.is_cuda(), "All input tensors must be CUDA tensors");
    TORCH_CHECK(c.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32,
                "All input tensors must be of type float32");
    TORCH_CHECK(c.dim() == 2, "Input tensor c must be 2D (B, d)");
    TORCH_CHECK(w.dim() == 2, "Weight tensor w must be 2D (d, out_dim)");
    TORCH_CHECK(b.dim() == 1, "Bias tensor b must be 1D (out_dim)");
    TORCH_CHECK(c.size(1) == w.size(0), "Inner dimensions of c and w must match (d)");
    TORCH_CHECK(w.size(1) == b.size(0), "Output dimension of w and b must match (out_dim)");

    const int B = c.size(0);
    const int d = c.size(1);
    const int out_dim = w.size(1);

    // --- Step 1: Apply SiLU to input tensor c ---
    // c_silu = SiLU(c)
    // This is an element-wise operation.
    auto c_silu = torch::empty_like(c); // Output tensor for SiLU(c)
    {
        const int threads_per_block_silu = 256;
        // Calculate number of blocks needed for a 1D grid-stride loop
        const int num_elements_silu = B * d;
        const int blocks_silu = (num_elements_silu + threads_per_block_silu - 1) / threads_per_block_silu;

        // Define the SiLU kernel inline (or could be a separate __global__ function)
        // This kernel is static to be defined only once.
        static __global__ void silu_elementwise_kernel(const float* in_ptr, float* out_ptr, int n_elements) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n_elements) {
                out_ptr[idx] = silu(in_ptr[idx]); // Use the __device__ silu function
            }
        }
        // Launch the SiLU kernel
        silu_elementwise_kernel<<<blocks_silu, threads_per_block_silu>>>(
            c.data_ptr<float>(),
            c_silu.data_ptr<float>(),
            num_elements_silu
        );
        // Synchronize to ensure SiLU computation is complete before cuBLAS call
        // In a single default stream, this might be implicit for subsequent CUDA calls on the same stream,
        // but explicit sync is safer, especially before cuBLAS or returning to CPU.
        cudaError_t err = cudaDeviceSynchronize();
        TORCH_CHECK(err == cudaSuccess, "CUDA error after silu_kernel: ", cudaGetErrorString(err));
    }

    // --- Step 2: Perform matrix multiplication: c_silu @ w ---
    // out_matmul = c_silu @ w
    // We use cuBLAS for optimized matrix multiplication.
    // PyTorch tensors are row-major. cuBLAS assumes column-major.
    // For C_rm = A_rm @ B_rm (rm: row-major)
    // This is equivalent to C_cm = (B_rm^T @ A_rm^T)^T (cm: column-major)
    // Let A_rm = c_silu [B, d], B_rm = w [d, out_dim]. Result C_rm = out [B, out_dim].
    // Then A_rm^T is [d, B] (A_prime_cm), B_rm^T is [out_dim, d] (B_prime_cm).
    // cublasSgemm(handle, op(B_prime_cm), op(A_prime_cm), m, n, k, ...)
    // where m=out_dim (rows of B_prime_cm), n=B (cols of A_prime_cm), k=d (common dim)
    // The matrices passed to cuBLAS are B_prime_cm (ptr to w) and A_prime_cm (ptr to c_silu)
    // if we want to use them as "op(matrix)"
    //
    // The current setup: cublasSgemm(handle, N, N, out_dim, B, d, w_ptr, c_silu_ptr)
    // A_cublas = w (ld = out_dim), interpreted as [out_dim, d] column-major
    // B_cublas = c_silu (ld = d), interpreted as [d, B] column-major
    // C_cublas = out (ld = out_dim), result is [out_dim, B] column-major
    // This correctly computes C_rm [B, out_dim] = c_silu_rm [B, d] @ w_rm [d, out_dim]
    auto out = torch::empty({B, out_dim}, c.options()); // Output tensor for the final result
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    TORCH_CHECK(stat == CUBLAS_STATUS_SUCCESS, "cuBLAS handle creation failed");

    const float alpha = 1.0f; // Multiplier for A*B
    const float beta = 0.0f;  // Multiplier for C before adding A*B (effectively out = alpha*A*B)

    // Note: For c_silu [B,d] and w [d,out_dim] -> out [B,out_dim]
    // The equivalent cuBLAS call to get out_col_major [out_dim, B] = w_col_major [out_dim,d] @ c_silu_col_major [d,B]
    // is correct. PyTorch's w[d,out_dim] (row-major) is w_col_major[out_dim,d] to cuBLAS.
    // PyTorch's c_silu[B,d] (row-major) is c_silu_col_major[d,B] to cuBLAS.
    stat = cublasSgemm(
        handle,
        CUBLAS_OP_N,        // Transposition for w: No transpose
        CUBLAS_OP_N,        // Transposition for c_silu: No transpose
        out_dim,            // m: rows of op(w) and op(out)
        B,                  // n: columns of op(c_silu) and op(out)
        d,                  // k: columns of op(w) and rows of op(c_silu)
        &alpha,
        w.data_ptr<float>(), out_dim, // w_col_major [out_dim, d], leading dimension out_dim
        c_silu.data_ptr<float>(), d,  // c_silu_col_major [d, B], leading dimension d
        &beta,
        out.data_ptr<float>(), out_dim // out_col_major [out_dim, B], leading dimension out_dim
    );
    TORCH_CHECK(stat == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed. Error: ", cublasGetStatusString(stat));

    // --- Step 3: Add bias b ---
    // out = out_matmul + b
    // This is an element-wise operation, where b is broadcasted.
    {
        const int total_elements_bias = B * out_dim;
        const int threads_per_block_bias = 256;
        const int blocks_bias = (total_elements_bias + threads_per_block_bias - 1) / threads_per_block_bias;

        static __global__ void add_bias_kernel(float* data_ptr, const float* bias_ptr, int n_elements, int feature_dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n_elements) {
                int channel_idx = idx % feature_dim; // Determine which bias element to add
                data_ptr[idx] += bias_ptr[channel_idx];
            }
        }
        // Launch the bias addition kernel
        add_bias_kernel<<<blocks_bias, threads_per_block_bias>>>(
            out.data_ptr<float>(),
            b.data_ptr<float>(),
            total_elements_bias,
            out_dim
        );
        cudaError_t err_bias = cudaDeviceSynchronize();
        TORCH_CHECK(err_bias == cudaSuccess, "CUDA error after add_bias_kernel: ", cudaGetErrorString(err_bias));
    }

    cublasDestroy(handle);
    return out;
}


// ============================================================================
// Adaptive Layer Normalization (adaLN)
// ============================================================================

/**
 * @brief CUDA kernel for Adaptive Layer Normalization.
 * For each vector x_vec of dimension C from the input x [B, T, C]:
 * 1. Computes mean and variance of x_vec.
 * 2. Normalizes x_vec: x_norm = (x_vec - mean) / sqrt(variance + eps).
 * 3. Applies learned scale and shift: y_vec = x_norm * scale_vec + shift_vec.
 * The scale and shift vectors are per-batch item (indexed by B) and applied to all T tokens.
 *
 * @param x Input tensor data of shape [B, T, C].
 * @param y Output tensor data of shape [B, T, C].
 * @param shift Shift parameters (beta) of shape [B, C].
 * @param scale Scale parameters (gamma) of shape [B, C].
 * @param B Batch size.
 * @param T Sequence length (number of tokens).
 * @param C Feature dimension (channels).
 * @param eps Epsilon for numerical stability in variance calculation.
 *
 * PERFORMANCE NOTE:
 * The current implementation calculates mean and variance *serially* within each thread
 * for the C dimension. This is highly inefficient for larger C. A production-grade
 * kernel would use parallel reduction (e.g., using shared memory) within each
 * CUDA block to compute mean and variance across the C dimension. Each block
 * would be responsible for one [T, C] slice or even one [C] vector if T=1 for LayerNorm.
 * The current launch strategy assigns one thread per [C] vector from the flattened [B*T, C] view.
 */
__global__ void ada_layernorm_kernel(
    const float* __restrict__ x,     // Input [B, T, C]
    float* __restrict__ y,           // Output [B, T, C]
    const float* __restrict__ shift, // Shift params (beta) [B, C]
    const float* __restrict__ scale, // Scale params (gamma) [B, C]
    int B,
    int T,
    int C,
    float eps
) {
    // Each thread processes one C-dimensional vector from the B*T total vectors.
    // Global thread index `idx` ranges from 0 to B*T - 1.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= B * T) { // Boundary check
        return;
    }

    // `b_t` is the index of the current token/vector being processed (from 0 to B*T - 1)
    int b_t = idx;

    // Pointer to the start of the current C-dimensional input vector
    const float* x_ptr = x + (long long)b_t * C;
    // Pointer to the start of the current C-dimensional output vector
    float* y_ptr = y + (long long)b_t * C;

    float current_mean = 0.0f;
    float current_var = 0.0f;

    // --- Calculate mean (serially per thread - INEFFICIENT for large C) ---
    // TODO: Optimize with parallel reduction using shared memory if C is large.
    for (int i = 0; i < C; ++i) {
        current_mean += x_ptr[i];
    }
    current_mean /= C;

    // --- Calculate variance (serially per thread - INEFFICIENT for large C) ---
    // TODO: Optimize with parallel reduction.
    for (int i = 0; i < C; ++i) {
        float diff = x_ptr[i] - current_mean;
        current_var += diff * diff;
    }
    current_var /= C;

    // Inverse standard deviation
    // rsqrtf is generally optimized to a fast hardware instruction.
    float inv_std = rsqrtf(current_var + eps);

    // Determine the batch index `b_idx` (from 0 to B-1) for this token.
    // This is used to select the correct shift and scale parameters.
    int b_idx = b_t / T; // Integer division gives the batch index

    // Pointers to the relevant shift and scale vectors for the current batch item
    const float* shift_vec_ptr = shift + (long long)b_idx * C;
    const float* scale_vec_ptr = scale + (long long)b_idx * C;

    // --- Apply normalization, scale, and shift ---
    for (int i = 0; i < C; ++i) {
        float normalized_val = (x_ptr[i] - current_mean) * inv_std;
        y_ptr[i] = normalized_val * scale_vec_ptr[i] + shift_vec_ptr[i];
    }
}

/**
 * @brief PyTorch frontend for the Adaptive Layer Normalization CUDA kernel.
 * This function is called from Python.
 *
 * @param x Input tensor of shape [B, T, C].
 * @param shift Shift parameters (beta) of shape [B, C], specific to each batch item.
 * @param scale Scale parameters (gamma) of shape [B, C], specific to each batch item.
 * @param eps Epsilon value for LayerNorm.
 * @return torch::Tensor Output tensor y of shape [B, T, C].
 */
torch::Tensor adaln_forward(
    torch::Tensor x,      // Input tensor [B, T, C]
    torch::Tensor shift,  // Shift parameters (beta) [B, C]
    torch::Tensor scale,  // Scale parameters (gamma) [B, C]
    float eps             // Epsilon for stability
) {
    // Input validation
    TORCH_CHECK(x.is_cuda() && shift.is_cuda() && scale.is_cuda(), "All input tensors must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && shift.dtype() == torch::kFloat32 && scale.dtype() == torch::kFloat32,
                "All input tensors must be of type float32");
    TORCH_CHECK(x.dim() == 3, "Input tensor x must be 3D (B, T, C)");
    TORCH_CHECK(shift.dim() == 2, "Shift tensor must be 2D (B, C)");
    TORCH_CHECK(scale.dim() == 2, "Scale tensor must be 2D (B, C)");
    TORCH_CHECK(x.size(0) == shift.size(0) && x.size(0) == scale.size(0), "Batch size B must match for x, shift, and scale");
    TORCH_CHECK(x.size(2) == shift.size(1) && x.size(2) == scale.size(1), "Channel size C must match for x, shift, and scale");

    const int B = x.size(0);
    const int T = x.size(1);
    const int C = x.size(2);

    // Allocate output tensor
    auto y = torch::empty_like(x);

    // Kernel launch configuration for ada_layernorm_kernel
    // We want B*T threads in total, each handling one C-dimensional vector.
    const int total_vectors = B * T;
    const int threads_per_block_adaln = 256; // A common choice, can be tuned
    const int num_blocks_adaln = (total_vectors + threads_per_block_adaln - 1) / threads_per_block_adaln;

    // Launch the adaLN kernel
    ada_layernorm_kernel<<<num_blocks_adaln, threads_per_block_adaln>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        shift.data_ptr<float>(),
        scale.data_ptr<float>(),
        B, T, C, eps
    );
    // Synchronize to ensure kernel completion before returning to PyTorch
    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after ada_layernorm_kernel: ", cudaGetErrorString(err));

    return y;
}

// ============================================================================
// Pybind11 Module Definition
// ============================================================================

/**
 * @brief Binds the C++/CUDA functions to a Python module.
 * TORCH_EXTENSION_NAME is a macro defined by the PyTorch build system (e.g., in setup.py).
 * This allows calling `your_module_name.silu_linear_forward(...)` from Python.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA implementations for SiLU+Linear and Adaptive LayerNorm"; // Optional module docstring
    m.def("silu_linear_forward", &silu_linear_forward, "SiLU + Linear forward (CUDA)");
    m.def("adaln_forward", &adaln_forward, "Adaptive Layer Normalization forward (CUDA)");
}