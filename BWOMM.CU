#include <cuda_runtime.h> // Required for CUDA types and calls

// --- Kernel Configuration ---
// Thread block dimensions
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16 // Results in 16x16 = 256 threads per block

// Work per thread (registers used for C_ij accumulation)
#define R_M 4 // Each thread computes R_M rows of the C tile
#define R_K 4 // Each thread computes R_K columns of the C tile

// Tile dimensions computed by a thread block for matrix C
#define TILE_M_C (BLOCK_DIM_Y * R_M) // 16 * 4 = 64
#define TILE_K_C (BLOCK_DIM_X * R_K) // 16 * 4 = 64

// Inner dimension for tiling (shared across A and B tiles)
#define TILE_N_SHARED 64 // This dimension will be iterated over by k_inner loop

// CUDA kernel for matrix multiplication (Highly Optimized Version)
__global__ void matrixMultiplicationKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    // Shared memory for tiles of A and B
    // As_s dimensions: TILE_M_C (rows from A) x TILE_N_SHARED (common dimension)
    // Bs_s dimensions: TILE_N_SHARED (common dimension) x TILE_K_C (cols from B)
    __shared__ float As_s[TILE_M_C][TILE_N_SHARED];
    __shared__ float Bs_s[TILE_N_SHARED][TILE_K_C];

    // Thread's local indices within the block
    int tx = threadIdx.x; // 0 to BLOCK_DIM_X - 1 (e.g., 0-15)
    int ty = threadIdx.y; // 0 to BLOCK_DIM_Y - 1 (e.g., 0-15)

    // Global base row and column for the C tile this block computes
    int block_C_row_base = blockIdx.y * TILE_M_C;
    int block_C_col_base = blockIdx.x * TILE_K_C;

    // Accumulators for the R_M x R_K C sub-tile computed by this thread
    float accum[R_M][R_K];

    // Initialize accumulators to 0.0f
    #pragma unroll
    for (int i = 0; i < R_M; ++i) {
        #pragma unroll
        for (int j = 0; j < R_K; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    // Loop over tiles in the N dimension (common dimension)
    int num_N_tiles = (N + TILE_N_SHARED - 1) / TILE_N_SHARED;
    for (int tile_n_idx = 0; tile_n_idx < num_N_tiles; ++tile_n_idx) {

        // --- Load tile of A into shared memory (As_s) ---
        // Each thread loads (TILE_M_C / BLOCK_DIM_Y) * (TILE_N_SHARED / BLOCK_DIM_X) elements.
        // (64/16) * (64/16) = 4 * 4 = 16 elements per thread for As_s.
        // These loops ensure coalesced global memory access.
        #pragma unroll
        for (int i_load_A = 0; i_load_A < (TILE_M_C / BLOCK_DIM_Y); ++i_load_A) { // Iterates R_M (4) times
            int s_load_A_row = ty + i_load_A * BLOCK_DIM_Y;
            int g_load_A_row = block_C_row_base + s_load_A_row;

            #pragma unroll
            for (int j_load_A = 0; j_load_A < (TILE_N_SHARED / BLOCK_DIM_X); ++j_load_A) { // Iterates TILE_N_SHARED/BLOCK_DIM_X (4) times
                int s_load_A_col = tx + j_load_A * BLOCK_DIM_X;
                int g_load_A_col = tile_n_idx * TILE_N_SHARED + s_load_A_col;

                if (g_load_A_row < M && g_load_A_col < N) {
                    As_s[s_load_A_row][s_load_A_col] = A[g_load_A_row * N + g_load_A_col];
                } else {
                    As_s[s_load_A_row][s_load_A_col] = 0.0f;
                }
            }
        }

        // --- Load tile of B into shared memory (Bs_s) ---
        // Each thread loads (TILE_N_SHARED / BLOCK_DIM_Y) * (TILE_K_C / BLOCK_DIM_X) elements.
        // (64/16) * (64/16) = 4 * 4 = 16 elements per thread for Bs_s.
        #pragma unroll
        for (int i_load_B = 0; i_load_B < (TILE_N_SHARED / BLOCK_DIM_Y); ++i_load_B) { // Iterates TILE_N_SHARED/BLOCK_DIM_Y (4) times
            int s_load_B_row = ty + i_load_B * BLOCK_DIM_Y;
            int g_load_B_row = tile_n_idx * TILE_N_SHARED + s_load_B_row;

            #pragma unroll
            for (int j_load_B = 0; j_load_B < (TILE_K_C / BLOCK_DIM_X); ++j_load_B) { // Iterates R_K (4) times
                int s_load_B_col = tx + j_load_B * BLOCK_DIM_X;
                int g_load_B_col = block_C_col_base + s_load_B_col;

                if (g_load_B_row < N && g_load_B_col < K) {
                    Bs_s[s_load_B_row][s_load_B_col] = B[g_load_B_row * K + g_load_B_col];
                } else {
                    Bs_s[s_load_B_row][s_load_B_col] = 0.0f;
                }
            }
        }
        
        __syncthreads(); // Ensure all data is loaded into shared memory

        // --- Compute matrix multiplication for this tile using shared memory ---
        // Iterate TILE_N_SHARED times (e.g., 64 times)
        #pragma unroll // Suggest unrolling for the k_inner loop if compiler deems beneficial
        for (int k_inner = 0; k_inner < TILE_N_SHARED; ++k_inner) {
            #pragma unroll
            for (int ir = 0; ir < R_M; ++ir) { // Loop over register tile rows (R_M = 4)
                // Row in As_s this thread processes for its C sub-tile's ir-th row
                int As_s_row_idx = ty * R_M + ir;
                float valA = As_s[As_s_row_idx][k_inner];
                
                #pragma unroll
                for (int ic = 0; ic < R_K; ++ic) { // Loop over register tile cols (R_K = 4)
                    // Col in Bs_s this thread processes for its C sub-tile's ic-th col
                    int Bs_s_col_idx = tx * R_K + ic;
                    float valB = Bs_s[k_inner][Bs_s_col_idx];
                    accum[ir][ic] += valA * valB;
                }
            }
        }
        __syncthreads(); // Ensure all computations for this tile are done before next load
    }

    // --- Store the computed R_M x R_K sub-tile to global memory C ---
    #pragma unroll
    for (int ir = 0; ir < R_M; ++ir) {
        int g_store_C_row = block_C_row_base + ty * R_M + ir;
        #pragma unroll
        for (int ic = 0; ic < R_K; ++ic) {
            int g_store_C_col = block_C_col_base + tx * R_K + ic;
            if (g_store_C_row < M && g_store_C_col < K) {
                C[g_store_C_row * K + g_store_C_col] = accum[ir][ic];
            }
        }
    }
}

// Solve function to compute matrix multiplication on GPU
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // Define block and grid dimensions
    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y); // e.g., 16x16 = 256 threads

    // Each block computes a TILE_M_C x TILE_K_C tile of C
    dim3 blocksPerGrid( (K + TILE_K_C - 1) / TILE_K_C,
                        (M + TILE_M_C - 1) / TILE_M_C );

    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
    // cudaError_t err = cudaGetLastError(); // Optional: error checking
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    // }

    cudaDeviceSynchronize();
}
