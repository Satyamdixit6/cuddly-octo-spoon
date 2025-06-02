#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

// CUDA kernel for 3D convolution (naive direct convolution)
// Assumes NCDHW format for input/output and KCDHW for weights
// K = number of output channels, C = number of input channels
// D, H, W = depth, height, width

__global__ void conv3d_forward_kernel(
    const float* input,      // Input tensor (N, C, D_in, H_in, W_in)
    const float* weights,    // Filter tensor (K, C, D_f, H_f, W_f)
    float* output,           // Output tensor (N, K, D_out, H_out, W_out)
    const float* bias,       // Bias tensor (K) - can be nullptr
    int N, int C, int D_in, int H_in, int W_in,
    int K, int D_f, int H_f, int W_f,
    int D_out, int H_out, int W_out,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {

    // Calculate output indices
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d_out_k_n_idx = blockIdx.z * blockDim.z + threadIdx.z; // Combined index for d_out, k, n

    // Decompose d_out_k_n_idx
    int d_out = d_out_k_n_idx % D_out;
    int k_n_idx = d_out_k_n_idx / D_out;
    int k = k_n_idx % K;
    int n = k_n_idx / K;


    if (w_out < W_out && h_out < H_out && d_out < D_out && k < K && n < N) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {          // Input channels
            for (int fd = 0; fd < D_f; ++fd) {   // Filter depth
                for (int fh = 0; fh < H_f; ++fh) { // Filter height
                    for (int fw = 0; fw < W_f; ++fw) { // Filter width
                        int d_in_orig = d_out * stride_d + fd - pad_d;
                        int h_in_orig = h_out * stride_h + fh - pad_h;
                        int w_in_orig = w_out * stride_w + fw - pad_w;

                        if (d_in_orig >= 0 && d_in_orig < D_in &&
                            h_in_orig >= 0 && h_in_orig < H_in &&
                            w_in_orig >= 0 && w_in_orig < W_in) {

                            // Input: N, C, D_in, H_in, W_in
                            long long input_idx = n * (C * D_in * H_in * W_in) +
                                                  c * (D_in * H_in * W_in) +
                                                  d_in_orig * (H_in * W_in) +
                                                  h_in_orig * W_in +
                                                  w_in_orig;

                            // Weights: K, C, D_f, H_f, W_f
                            long long weight_idx = k * (C * D_f * H_f * W_f) +
                                                   c * (D_f * H_f * W_f) +
                                                   fd * (H_f * W_f) +
                                                   fh * W_f +
                                                   fw;
                            acc += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
            }
        }
        if (bias != nullptr) {
            acc += bias[k];
        }

        // Output: N, K, D_out, H_out, W_out
        long long output_idx = n * (K * D_out * H_out * W_out) +
                               k * (D_out * H_out * W_out) +
                               d_out * (H_out * W_out) +
                               h_out * W_out +
                               w_out;
        output[output_idx] = acc;
    }
}

// Wrapper function to launch the kernel
// This is what you'll call from the C++ binding
extern "C" cudaError_t conv3d_forward_cuda_launcher(
    const float* input, const float* weights, float* output, const float* bias,
    int N, int C, int D_in, int H_in, int W_in,
    int K, int D_f, int H_f, int W_f,
    int D_out, int H_out, int W_out,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {

    dim3 threadsPerBlock(16, 16, 1); // Adjust for optimal performance
    // Maximize Z dimension of grid for combined index (d_out * K * N)
    // Max grid dim for Z is 65535
    dim3 numBlocks(
        (W_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (H_out + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (D_out * K * N + threadsPerBlock.z -1) / threadsPerBlock.z
    );
    
    if (numBlocks.z > 65535) {
        // This indicates the problem size is too large for this simple Z-dimension grid decomposition.
        // A more robust kernel launch configuration or kernel decomposition would be needed.
        // For now, we'll just print an error. A real solution might involve multiple kernel launches
        // or a different indexing scheme.
        fprintf(stderr, "Error: Z dimension of grid (%u) exceeds maximum (65535).\n", numBlocks.z);
        fprintf(stderr, "N=%d, K=%d, D_out=%d. Total Z items: %lld\n", N, K, D_out, (long long)D_out * K * N);

        // Fallback or error handling:
        // One simple (but not necessarily optimal) fallback is to cap numBlocks.z and iterate inside the kernel,
        // or to launch multiple kernels. Here, we'll just return an error.
        // Or, if N=1, K is large, D_out is large. We might need to make threadsPerBlock.z larger
        // and reduce threadsPerBlock.x and .y, or use a 2D grid for k_n_idx and 1D for d_out.
        // This kernel launch configuration needs careful thought for large dimensions.
        return cudaErrorInvalidConfiguration;
    }


    conv3d_forward_kernel<<<numBlocks, threadsPerBlock>>>(
        input, weights, output, bias,
        N, C, D_in, H_in, W_in,
        K, D_f, H_f, W_f,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );

    return cudaGetLastError();
}