#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call)                                                  \
do {                                                                      \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));             \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

// Network Dimensions and Hyperparameters
const int N_INPUT = 4;    // Number of input features
const int N_HIDDEN = 8;   // Number of neurons in the hidden layer
const int N_OUTPUT = 1;   // Number of output neurons (1 for binary classification)
const int BATCH_SIZE = 1; // Batch size (simplified to 1 for this example)

const float LEARNING_RATE = 0.01f;
const int EPOCHS = 1000;
const float EPSILON = 1e-7f; // For numerical stability in log

// Threads per block for kernel launches
const int TPB = 256; // For 1D kernels
const int TPB_2D = 16; // For 2D kernels (TPB_2D x TPB_2D threads)

// ---------------- KERNEL DEFINITIONS ----------------

// Kernel for Matrix Multiplication: C = A * B (NN: No Transpose, No Transpose)
// A: M x K, B: K x N, C: M x N
__global__ void gemm_NN_kernel(float* C, const float* A, const float* B, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel for Matrix Multiplication: C = A^T * B (TN: Transpose A, No Transpose B)
// A_orig: K x M (original A before transpose), B_orig: K x N, C: M x N
// This computes C[m][n] = sum_k A_orig[k][m] * B_orig[k][n]
// A_orig_rows = K (rows of A_orig, common dimension for sum)
// A_orig_cols = M (cols of A_orig, rows of C)
// B_orig_cols = N (cols of B_orig, cols of C)
__global__ void gemm_TN_kernel(float* C, const float* A_orig, const float* B_orig,
                               int A_orig_rows, int A_orig_cols, int B_orig_cols) {
    int c_row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension (0 to A_orig_cols-1)
    int c_col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension (0 to B_orig_cols-1)

    if (c_row < A_orig_cols && c_col < B_orig_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_orig_rows; ++k) { // Sum over K (A_orig_rows)
            sum += A_orig[k * A_orig_cols + c_row] * B_orig[k * B_orig_cols + c_col];
        }
        C[c_row * B_orig_cols + c_col] = sum;
    }
}


// Kernel for Matrix Multiplication: C = A * B^T (NT: No Transpose A, Transpose B)
// A_orig: M x K, B_orig: N x K (original B before transpose), C: M x N
// This computes C[m][n] = sum_k A_orig[m][k] * B_orig[n][k]
// A_orig_rows = M (rows of A_orig, rows of C)
// A_orig_cols = K (cols of A_orig, common dimension for sum)
// B_orig_rows_for_B_T = N (rows of B_orig, cols of C)
__global__ void gemm_NT_kernel(float* C, const float* A_orig, const float* B_orig,
                               int A_orig_rows, int A_orig_cols, int B_orig_rows_for_B_T) {
    int c_row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension (0 to A_orig_rows-1)
    int c_col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension (0 to B_orig_rows_for_B_T-1)

    if (c_row < A_orig_rows && c_col < B_orig_rows_for_B_T) {
        float sum = 0.0f;
        for (int k = 0; k < A_orig_cols; ++k) { // Sum over K (A_orig_cols)
            sum += A_orig[c_row * A_orig_cols + k] * B_orig[c_col * A_orig_cols + k];
        }
        C[c_row * B_orig_rows_for_B_T + c_col] = sum;
    }
}


// Kernel to add bias vector to a matrix: Z = Z_no_bias + bias
// Z_matrix: R x C, bias_vector: 1 x C
__global__ void add_bias_kernel(float* Z_matrix, const float* bias_vector, int R, int C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < R && col < C) {
        Z_matrix[row * C + col] += bias_vector[col];
    }
}

// ReLU forward activation: A = max(0, Z)
__global__ void relu_forward_kernel(float* A, const float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmaxf(0.0f, Z[idx]);
    }
}

// Sigmoid forward activation: A = 1 / (1 + exp(-Z))
__global__ void sigmoid_forward_kernel(float* A, const float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = 1.0f / (1.0f + expf(-Z[idx]));
    }
}

// ReLU backward: dZ = dA * (Z > 0 ? 1 : 0)
__global__ void relu_backward_kernel(float* dZ, const float* dA, const float* Z_fwd, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = dA[idx] * (Z_fwd[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

// SGD update: params = params - learning_rate * grads
__global__ void sgd_update_kernel(float* params, const float* grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= lr * grads[idx];
    }
}

// Compute dZ for the output layer (Sigmoid + BCE Loss): dZ2 = A2 - y_true
__global__ void compute_dZ2_kernel(float* dZ2, const float* A2, const float* y_true, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ2[idx] = A2[idx] - y_true[idx];
    }
}

// Copy kernel (useful for db = dZ when batch_size = 1)
__global__ void copy_kernel(float* dest, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dest[idx] = src[idx];
    }
}


// ---------------- NEURAL NETWORK STRUCTURE ----------------
typedef struct {
    // Parameters (device pointers)
    float *d_W1, *d_b1; // Layer 1 (Input -> Hidden)
    float *d_W2, *d_b2; // Layer 2 (Hidden -> Output)

    // Gradients (device pointers)
    float *d_dW1, *d_db1;
    float *d_dW2, *d_db2;

    // Activations and intermediate values for forward/backward pass (device pointers)
    float *d_X;         // Input (BATCH_SIZE x N_INPUT)
    float *d_Z1_no_bias; // Linear output 1 before bias (BATCH_SIZE x N_HIDDEN)
    float *d_Z1;        // Linear output 1 after bias (BATCH_SIZE x N_HIDDEN)
    float *d_A1;        // ReLU output (BATCH_SIZE x N_HIDDEN)
    float *d_Z2_no_bias; // Linear output 2 before bias (BATCH_SIZE x N_OUTPUT)
    float *d_Z2;        // Linear output 2 after bias (BATCH_SIZE x N_OUTPUT)
    float *d_A2;        // Sigmoid output (BATCH_SIZE x N_OUTPUT)
    
    float *d_y_true;    // True labels (BATCH_SIZE x N_OUTPUT)

    // Gradients of loss w.r.t. Z and A (device pointers)
    float *d_dZ2;       // Gradient dLoss/dZ2
    float *d_dA1;       // Gradient dLoss/dA1
    float *d_dZ1;       // Gradient dLoss/dZ1

} NeuralNetworkCUDA;

// ---------------- HOST HELPER FUNCTIONS ----------------

// Initialize network parameters (weights and biases) on the host
void init_network_params_on_host(float** h_W1, float** h_b1, float** h_W2, float** h_b2) {
    *h_W1 = (float*)malloc(N_INPUT * N_HIDDEN * sizeof(float));
    *h_b1 = (float*)malloc(N_HIDDEN * sizeof(float)); // Bias is 1 x N_HIDDEN
    *h_W2 = (float*)malloc(N_HIDDEN * N_OUTPUT * sizeof(float));
    *h_b2 = (float*)malloc(N_OUTPUT * sizeof(float)); // Bias is 1 x N_OUTPUT

    srand(time(0)); // Seed for random number generation

    // Initialize W1, b1 (e.g., Xavier/He initialization or simple scaled random)
    for (int i = 0; i < N_INPUT * N_HIDDEN; ++i) (*h_W1)[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f; // Small random weights
    for (int i = 0; i < N_HIDDEN; ++i) (*h_b1)[i] = 0.0f; // Zero biases

    // Initialize W2, b2
    for (int i = 0; i < N_HIDDEN * N_OUTPUT; ++i) (*h_W2)[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < N_OUTPUT; ++i) (*h_b2)[i] = 0.0f;
}

// Allocate memory for the neural network on the device
void allocate_network_on_device(NeuralNetworkCUDA* net) {
    // Parameters
    CUDA_CHECK(cudaMalloc((void**)&net->d_W1, N_INPUT * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_b1, N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_W2, N_HIDDEN * N_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_b2, N_OUTPUT * sizeof(float)));

    // Gradients
    CUDA_CHECK(cudaMalloc((void**)&net->d_dW1, N_INPUT * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_db1, N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_dW2, N_HIDDEN * N_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_db2, N_OUTPUT * sizeof(float)));

    // Activations and intermediate values
    CUDA_CHECK(cudaMalloc((void**)&net->d_X, BATCH_SIZE * N_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_Z1_no_bias, BATCH_SIZE * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_Z1, BATCH_SIZE * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_A1, BATCH_SIZE * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_Z2_no_bias, BATCH_SIZE * N_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_Z2, BATCH_SIZE * N_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_A2, BATCH_SIZE * N_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_y_true, BATCH_SIZE * N_OUTPUT * sizeof(float)));
    
    // Gradients of loss w.r.t. Z and A
    CUDA_CHECK(cudaMalloc((void**)&net->d_dZ2, BATCH_SIZE * N_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_dA1, BATCH_SIZE * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_dZ1, BATCH_SIZE * N_HIDDEN * sizeof(float)));
}

// Copy initial parameters from host to device
void copy_params_to_device(NeuralNetworkCUDA* net, float* h_W1, float* h_b1, float* h_W2, float* h_b2) {
    CUDA_CHECK(cudaMemcpy(net->d_W1, h_W1, N_INPUT * N_HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b1, h_b1, N_HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, h_W2, N_HIDDEN * N_OUTPUT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b2, h_b2, N_OUTPUT * sizeof(float), cudaMemcpyHostToDevice));
}

// Free all allocated device memory
void free_network_on_device(NeuralNetworkCUDA* net) {
    cudaFree(net->d_W1); cudaFree(net->d_b1); cudaFree(net->d_W2); cudaFree(net->d_b2);
    cudaFree(net->d_dW1); cudaFree(net->d_db1); cudaFree(net->d_dW2); cudaFree(net->d_db2);
    cudaFree(net->d_X); cudaFree(net->d_Z1_no_bias); cudaFree(net->d_Z1); cudaFree(net->d_A1);
    cudaFree(net->d_Z2_no_bias); cudaFree(net->d_Z2); cudaFree(net->d_A2); cudaFree(net->d_y_true);
    cudaFree(net->d_dZ2); cudaFree(net->d_dA1); cudaFree(net->d_dZ1);
}

// Free host parameter memory
void free_network_params_on_host(float* h_W1, float* h_b1, float* h_W2, float* h_b2) {
    free(h_W1); free(h_b1); free(h_W2); free(h_b2);
}

// ---------------- CORE NEURAL NETWORK OPERATIONS ----------------

void forward_pass(NeuralNetworkCUDA* net, const float* h_X_batch) {
    // Copy current batch input to device
    CUDA_CHECK(cudaMemcpy(net->d_X, h_X_batch, BATCH_SIZE * N_INPUT * sizeof(float), cudaMemcpyHostToDevice));

    // Layer 1: Z1 = X * W1 + b1
    // Z1_no_bias = X * W1
    dim3 blocks_gemm1((N_HIDDEN + TPB_2D - 1) / TPB_2D, (BATCH_SIZE + TPB_2D - 1) / TPB_2D);
    dim3 threads_gemm(TPB_2D, TPB_2D);
    gemm_NN_kernel<<<blocks_gemm1, threads_gemm>>>(net->d_Z1_no_bias, net->d_X, net->d_W1, BATCH_SIZE, N_INPUT, N_HIDDEN);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy Z1_no_bias to Z1 to prepare for bias addition
    CUDA_CHECK(cudaMemcpy(net->d_Z1, net->d_Z1_no_bias, BATCH_SIZE * N_HIDDEN * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Z1 = Z1_no_bias + b1
    dim3 blocks_add_bias1((N_HIDDEN + TPB_2D - 1) / TPB_2D, (BATCH_SIZE + TPB_2D - 1) / TPB_2D);
    add_bias_kernel<<<blocks_add_bias1, threads_gemm>>>(net->d_Z1, net->d_b1, BATCH_SIZE, N_HIDDEN);
    CUDA_CHECK(cudaGetLastError());

    // A1 = ReLU(Z1)
    dim3 blocks_relu((BATCH_SIZE * N_HIDDEN + TPB - 1) / TPB, 1);
    dim3 threads_relu(TPB, 1);
    relu_forward_kernel<<<blocks_relu, threads_relu>>>(net->d_A1, net->d_Z1, BATCH_SIZE * N_HIDDEN);
    CUDA_CHECK(cudaGetLastError());

    // Layer 2: Z2 = A1 * W2 + b2
    // Z2_no_bias = A1 * W2
    dim3 blocks_gemm2((N_OUTPUT + TPB_2D - 1) / TPB_2D, (BATCH_SIZE + TPB_2D - 1) / TPB_2D);
    gemm_NN_kernel<<<blocks_gemm2, threads_gemm>>>(net->d_Z2_no_bias, net->d_A1, net->d_W2, BATCH_SIZE, N_HIDDEN, N_OUTPUT);
    CUDA_CHECK(cudaGetLastError());

    // Copy Z2_no_bias to Z2
    CUDA_CHECK(cudaMemcpy(net->d_Z2, net->d_Z2_no_bias, BATCH_SIZE * N_OUTPUT * sizeof(float), cudaMemcpyDeviceToDevice));

    // Z2 = Z2_no_bias + b2
    dim3 blocks_add_bias2((N_OUTPUT + TPB_2D - 1) / TPB_2D, (BATCH_SIZE + TPB_2D - 1) / TPB_2D);
    add_bias_kernel<<<blocks_add_bias2, threads_gemm>>>(net->d_Z2, net->d_b2, BATCH_SIZE, N_OUTPUT);
    CUDA_CHECK(cudaGetLastError());
    
    // A2 = Sigmoid(Z2)
    dim3 blocks_sigmoid((BATCH_SIZE * N_OUTPUT + TPB - 1) / TPB, 1);
    sigmoid_forward_kernel<<<blocks_sigmoid, threads_relu>>>(net->d_A2, net->d_Z2, BATCH_SIZE * N_OUTPUT);
    CUDA_CHECK(cudaGetLastError());
}

// Computes BCE loss and dZ2 (dLoss/dZ2 = A2 - y_true)
float compute_loss_and_dZ2(NeuralNetworkCUDA* net, const float* h_y_true_batch) {
    // Copy true labels to device
    CUDA_CHECK(cudaMemcpy(net->d_y_true, h_y_true_batch, BATCH_SIZE * N_OUTPUT * sizeof(float), cudaMemcpyHostToDevice));

    // Compute dZ2 = A2 - y_true on device
    dim3 blocks_dz2((BATCH_SIZE * N_OUTPUT + TPB - 1) / TPB, 1);
    dim3 threads_dz2(TPB, 1);
    compute_dZ2_kernel<<<blocks_dz2, threads_dz2>>>(net->d_dZ2, net->d_A2, net->d_y_true, BATCH_SIZE * N_OUTPUT);
    CUDA_CHECK(cudaGetLastError());

    // For loss calculation, copy A2 and y_true back to host (for BATCH_SIZE=1, N_OUTPUT=1)
    // For larger batches/outputs, a reduction kernel for loss would be better.
    float h_A2[BATCH_SIZE * N_OUTPUT];
    float h_y_true[BATCH_SIZE * N_OUTPUT];
    CUDA_CHECK(cudaMemcpy(h_A2, net->d_A2, BATCH_SIZE * N_OUTPUT * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y_true, net->d_y_true, BATCH_SIZE * N_OUTPUT * sizeof(float), cudaMemcpyDeviceToHost));
    
    float loss = 0.0f;
    for (int i = 0; i < BATCH_SIZE * N_OUTPUT; ++i) {
        loss -= (h_y_true[i] * logf(h_A2[i] + EPSILON) + (1.0f - h_y_true[i]) * logf(1.0f - h_A2[i] + EPSILON));
    }
    return loss / (BATCH_SIZE * N_OUTPUT); // Average loss
}

void backward_pass(NeuralNetworkCUDA* net) {
    // dZ2 is already computed in compute_loss_and_dZ2

    // --- Layer 2 (Output Layer) Gradients ---
    // dW2 = A1^T * dZ2
    // A1 is BATCH_SIZE x N_HIDDEN. dZ2 is BATCH_SIZE x N_OUTPUT. dW2 is N_HIDDEN x N_OUTPUT.
    // A_orig = A1 (BATCH_SIZE x N_HIDDEN), B_orig = dZ2 (BATCH_SIZE x N_OUTPUT)
    // A_orig_rows = BATCH_SIZE, A_orig_cols = N_HIDDEN, B_orig_cols = N_OUTPUT
    // C_rows = N_HIDDEN, C_cols = N_OUTPUT
    dim3 blocks_dW2((N_OUTPUT + TPB_2D - 1) / TPB_2D, (N_HIDDEN + TPB_2D - 1) / TPB_2D);
    dim3 threads_gemm(TPB_2D, TPB_2D);
    gemm_TN_kernel<<<blocks_dW2, threads_gemm>>>(net->d_dW2, net->d_A1, net->d_dZ2,
                                                BATCH_SIZE, N_HIDDEN, N_OUTPUT);
    CUDA_CHECK(cudaGetLastError());

    // db2 = sum(dZ2) over batch. For BATCH_SIZE=1, db2 = dZ2.
    // For BATCH_SIZE > 1, a reduction kernel would be needed.
    dim3 blocks_db2((N_OUTPUT + TPB - 1) / TPB, 1);
    dim3 threads_db(TPB, 1);
    copy_kernel<<<blocks_db2, threads_db>>>(net->d_db2, net->d_dZ2, N_OUTPUT); // Assuming BATCH_SIZE=1
    CUDA_CHECK(cudaGetLastError());

    // dA1 = dZ2 * W2^T
    // dZ2 is BATCH_SIZE x N_OUTPUT. W2 is N_HIDDEN x N_OUTPUT. dA1 is BATCH_SIZE x N_HIDDEN.
    // A_orig = dZ2 (BATCH_SIZE x N_OUTPUT), B_orig = W2 (N_HIDDEN x N_OUTPUT)
    // A_orig_rows = BATCH_SIZE, A_orig_cols = N_OUTPUT, B_orig_rows_for_B_T = N_HIDDEN
    // C_rows = BATCH_SIZE, C_cols = N_HIDDEN
    dim3 blocks_dA1((N_HIDDEN + TPB_2D - 1) / TPB_2D, (BATCH_SIZE + TPB_2D - 1) / TPB_2D);
    gemm_NT_kernel<<<blocks_dA1, threads_gemm>>>(net->d_dA1, net->d_dZ2, net->d_W2,
                                                BATCH_SIZE, N_OUTPUT, N_HIDDEN);
    CUDA_CHECK(cudaGetLastError());

    // --- Layer 1 (Hidden Layer) Gradients ---
    // dZ1 = dA1 * ReLU_derivative(Z1)
    dim3 blocks_dZ1((BATCH_SIZE * N_HIDDEN + TPB - 1) / TPB, 1);
    relu_backward_kernel<<<blocks_dZ1, threads_db>>>(net->d_dZ1, net->d_dA1, net->d_Z1, BATCH_SIZE * N_HIDDEN);
    CUDA_CHECK(cudaGetLastError());

    // dW1 = X^T * dZ1
    // X is BATCH_SIZE x N_INPUT. dZ1 is BATCH_SIZE x N_HIDDEN. dW1 is N_INPUT x N_HIDDEN.
    // A_orig = X (BATCH_SIZE x N_INPUT), B_orig = dZ1 (BATCH_SIZE x N_HIDDEN)
    // A_orig_rows = BATCH_SIZE, A_orig_cols = N_INPUT, B_orig_cols = N_HIDDEN
    // C_rows = N_INPUT, C_cols = N_HIDDEN
    dim3 blocks_dW1((N_HIDDEN + TPB_2D - 1) / TPB_2D, (N_INPUT + TPB_2D - 1) / TPB_2D);
    gemm_TN_kernel<<<blocks_dW1, threads_gemm>>>(net->d_dW1, net->d_X, net->d_dZ1,
                                                BATCH_SIZE, N_INPUT, N_HIDDEN);
    CUDA_CHECK(cudaGetLastError());

    // db1 = sum(dZ1) over batch. For BATCH_SIZE=1, db1 = dZ1.
    dim3 blocks_db1((N_HIDDEN + TPB - 1) / TPB, 1);
    copy_kernel<<<blocks_db1, threads_db>>>(net->d_db1, net->d_dZ1, N_HIDDEN); // Assuming BATCH_SIZE=1
    CUDA_CHECK(cudaGetLastError());
}

void optimize(NeuralNetworkCUDA* net, float learning_rate) {
    dim3 threads_sgd(TPB, 1);

    // Update W1, b1
    dim3 blocks_W1((N_INPUT * N_HIDDEN + TPB - 1) / TPB, 1);
    sgd_update_kernel<<<blocks_W1, threads_sgd>>>(net->d_W1, net->d_dW1, learning_rate, N_INPUT * N_HIDDEN);
    CUDA_CHECK(cudaGetLastError());

    dim3 blocks_b1((N_HIDDEN + TPB - 1) / TPB, 1);
    sgd_update_kernel<<<blocks_b1, threads_sgd>>>(net->d_b1, net->d_db1, learning_rate, N_HIDDEN);
    CUDA_CHECK(cudaGetLastError());

    // Update W2, b2
    dim3 blocks_W2((N_HIDDEN * N_OUTPUT + TPB - 1) / TPB, 1);
    sgd_update_kernel<<<blocks_W2, threads_sgd>>>(net->d_W2, net->d_dW2, learning_rate, N_HIDDEN * N_OUTPUT);
    CUDA_CHECK(cudaGetLastError());

    dim3 blocks_b2((N_OUTPUT + TPB - 1) / TPB, 1);
    sgd_update_kernel<<<blocks_b2, threads_sgd>>>(net->d_b2, net->d_db2, learning_rate, N_OUTPUT);
    CUDA_CHECK(cudaGetLastError());
}


// ---------------- MAIN FUNCTION (TRAINING AND TEST) ----------------
int main() {
    CUDA_CHECK(cudaSetDevice(0)); // Select GPU device

    NeuralNetworkCUDA net;
    float *h_W1, *h_b1, *h_W2, *h_b2;

    // 1. Initialize parameters on host
    init_network_params_on_host(&h_W1, &h_b1, &h_W2, &h_b2);

    // 2. Allocate memory on device
    allocate_network_on_device(&net);

    // 3. Copy initial parameters to device
    copy_params_to_device(&net, h_W1, h_b1, h_W2, h_b2);

    // Sample data (XOR-like problem for N_INPUT=2, extend for N_INPUT=4)
    // For N_INPUT = 4, let's create a simple pattern
    float h_X_sample[N_INPUT];
    float h_y_sample[N_OUTPUT];

    printf("Training the neural network...\n");
    printf("Epoch | Loss\n");
    printf("------|------\n");

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Create a synthetic sample for each epoch (or use a fixed dataset)
        // Example: if first two inputs are high, output is 1, else 0
        for(int i=0; i < N_INPUT; ++i) h_X_sample[i] = (float)rand() / RAND_MAX;

        if (h_X_sample[0] > 0.5f && h_X_sample[1] > 0.5f) {
             h_y_sample[0] = 1.0f;
        } else {
             h_y_sample[0] = 0.0f;
        }
        // If N_INPUT > 2, this logic might need to be more complex for a meaningful task
        // For N_INPUT=4, let's make it if sum of first 2 > sum of last 2
        if (N_INPUT == 4) {
            if ((h_X_sample[0] + h_X_sample[1]) > (h_X_sample[2] + h_X_sample[3])) {
                h_y_sample[0] = 1.0f;
            } else {
                h_y_sample[0] = 0.0f;
            }
        }


        // Perform one training step
        forward_pass(&net, h_X_sample);
        float loss = compute_loss_and_dZ2(&net, h_y_sample);
        backward_pass(&net);
        optimize(&net, LEARNING_RATE);

        if ((epoch + 1) % (EPOCHS / 10) == 0 || epoch == 0) {
            printf("%5d | %.4f\n", epoch + 1, loss);
        }
    }

    printf("\nTraining finished.\n");

    // Test with a sample input after training
    printf("\nTesting with a sample input:\n");
    // Create a test sample
    float h_X_test[N_INPUT];
    for(int i=0; i < N_INPUT; ++i) h_X_test[i] = (float)rand() / RAND_MAX;
    
    float true_label_test;
     if (N_INPUT == 4) { // Same logic as training for consistency
        if ((h_X_test[0] + h_X_test[1]) > (h_X_test[2] + h_X_test[3])) {
            true_label_test = 1.0f;
        } else {
            true_label_test = 0.0f;
        }
    } else { // Fallback for N_INPUT != 4
        if (h_X_test[0] > 0.5f && h_X_test[1] > 0.5f) {
             true_label_test = 1.0f;
        } else {
             true_label_test = 0.0f;
        }
    }


    printf("Input: [");
    for(int i=0; i<N_INPUT; ++i) printf("%.2f ", h_X_test[i]);
    printf("]\n");

    forward_pass(&net, h_X_test); // Run forward pass with test data

    float h_A2_test[N_OUTPUT * BATCH_SIZE];
    CUDA_CHECK(cudaMemcpy(h_A2_test, net.d_A2, N_OUTPUT * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Predicted output (A2): %.4f\n", h_A2_test[0]);
    printf("True label: %.1f\n", true_label_test);
    printf("Predicted label (threshold 0.5): %d\n", h_A2_test[0] > 0.5f ? 1 : 0);


    // Cleanup
    free_network_on_device(&net);
    free_network_params_on_host(h_W1, h_b1, h_W2, h_b2);

    CUDA_CHECK(cudaDeviceReset()); // Reset device
    return 0;
}

