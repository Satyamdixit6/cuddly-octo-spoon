
 #include <cute/tensor.hpp>
 #include <cutlass/cutlass.h>
 #include <cutlass/array.h>
 #include <cutlass/numeric_types.h>
 #include <cuda_runtime.h>
 #include <stdio.h>
 
 // Use CuTe and CUTLASS namespaces
 using namespace cute;
 using namespace cutlass;
 
 // Kernel traits with real types
 template<typename T>
 struct Kernel_traits {
     using Element = T;
     using ElementAccum = float;
     static constexpr int kBlockM = 4;      // Block size for query
     static constexpr int kBlockN = 4;      // Block size for key/value
     static constexpr int kHeadDim = 8;     // Head dimension
     static constexpr int kThreads = 32;    // Threads per block
     static constexpr int kMinBlocksPerSm = 1;
     static constexpr int kNWarps = kThreads / 32;
     using TiledMma = typename cutlass::gemm::threadblock::DefaultMma<T, cutlass::arch::Sm80, 
                              cutlass::gemm::GemmShape<16, 8, 8>, 1>::ThreadblockMma;
     using SmemLayoutQ = typename cutlass::layout::RowMajor;
     using SmemLayoutKV = typename cutlass::layout::RowMajor;
     using SmemLayoutVtransposed = typename cutlass::layout::ColumnMajor;
     using SmemCopyAtom = cutlass::Copy_Atom<cutlass::universal::DefaultCopy, T>;
     using SmemCopyAtomO = cutlass::Copy_Atom<cutlass::universal::DefaultCopy, T>;
 };
 
 // Parameters struct
 struct Params {
     float* o_ptr;              // Output pointer
     float* k_ptr;              // Key pointer
     float* v_ptr;              // Value pointer
     float* softmax_lse_ptr;    // Log-sum-exp pointer
     int b;                     // Batch size
     int h;                     // Number of heads
     int d;                     // Head dimension
     int seqlen_q;              // Query sequence length
     int seqlen_k;              // Key sequence length
     int total_q;               // Total query length
     int o_batch_stride;
     int o_row_stride;
     int o_head_stride;
     int k_batch_stride;
     int k_row_stride;
     int k_head_stride;
     int v_batch_stride;
     int v_row_stride;
     int v_head_stride;
     float scale_softmax_log2;
     float softcap;
     bool unpadded_lse;
     bool seqlenq_ngroups_swapped;
     int window_size_left;
     int window_size_right;
     uint64_t philox_args[2];   // For dropout
     uint8_t p_dropout_in_uint8_t;
     int h_h_k_ratio;
     float* alibi_slopes_ptr;
     int alibi_slopes_batch_stride;
 };
 
 // BlockInfo for sequence length handling
 template<bool Varlen>
 struct BlockInfo {
     int actual_seqlen_q;
     int actual_seqlen_k;
     __device__ __host__ BlockInfo(const Params& params, int bidb) 
         : actual_seqlen_q(params.seqlen_q), actual_seqlen_k(params.seqlen_k) {}
     __device__ int q_offset(int batch_stride, int row_stride, int bidb) const { 
         return bidb * batch_stride; 
     }
     __device__ int k_offset(int batch_stride, int row_stride, int bidb) const { 
         return bidb * batch_stride; 
     }
 };
 
 // Function to get output tile using CuTe tensors
 template<typename Element, typename Params, int kBlockM, int kHeadDim, bool Is_even_MN>
 __forceinline__ __device__ auto get_output_tile(
     const Params& params,
     const BlockInfo<Is_even_MN>& binfo,
     int bidb,
     int bidh,
     int m_block
 ) {
     Element* base_ptr = params.o_ptr + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
     auto mO = make_tensor(make_gmem_ptr(base_ptr), make_shape(binfo.actual_seqlen_q, params.h, params.d),
                           make_stride(params.o_row_stride, params.o_head_stride, 1));
     return local_tile(mO(_, bidh, _), make_shape(kBlockM, kHeadDim), make_coord(m_block, 0));
 }
 
 // Simplified attention kernel
 template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, 
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
 __launch_bounds__(Kernel_traits::kThreads, Kernel_traits::kMinBlocksPerSm)
 __global__ void compute_attn_kernel(const Params params) {
     using Element = typename Kernel_traits::Element;
     const int bidb = blockIdx.y;  // Batch index
     const int bidh = blockIdx.z;  // Head index
     const int m_block = blockIdx.x;  // Query block index
 
     extern __shared__ float smem_[];
     constexpr int kBlockM = Kernel_traits::kBlockM;
     constexpr int kHeadDim = Kernel_traits::kHeadDim;
 
     BlockInfo<!Is_even_MN> binfo(params, bidb);
     if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
 
     // Shared memory tensor
     auto acc_o = make_tensor(make_smem_ptr(smem_), make_shape(kBlockM, kHeadDim), make_layout(kHeadDim));
     clear(acc_o);  // Initialize to zero
 
     // Get output tile and copy to global memory
     auto gO_final = get_output_tile<Element, Params, kBlockM, kHeadDim, Is_even_MN>(params, binfo, bidb, bidh, m_block);
     copy(acc_o, gO_final);
 }
 
 // Launch function for Flash Attention
 template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi,
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
 void flash_attention(const Params& params) {
     dim3 grid((params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM, params.b, params.h);
     dim3 block(Kernel_traits::kThreads);
     size_t smem_size = Kernel_traits::kBlockM * Kernel_traits::kHeadDim * sizeof(float);
 
     compute_attn_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                         Is_even_MN, Is_even_K, Is_softcap, Return_softmax>
         <<<grid, block, smem_size>>>(params);
 
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("CUDA error: %s\n", cudaGetErrorString(err));
     }
     cudaDeviceSynchronize();
 }
 
 // Main function to test the implementation
 int main() {
     using KT = Kernel_traits<float>;
 
     // Problem dimensions
     const int batch_size = 2;
     const int num_heads = 2;
     const int seqlen_q = 8;
     const int seqlen_k = 8;
     const int head_dim = KT::kHeadDim;
 
     // Allocate device memory
     float* d_output;
     cudaMalloc(&d_output, batch_size * seqlen_q * num_heads * head_dim * sizeof(float));
     cudaMemset(d_output, 0, batch_size * seqlen_q * num_heads * head_dim * sizeof(float));
 
     // Initialize Params
     Params params;
     params.o_ptr = d_output;
     params.k_ptr = nullptr;  // Not used in this simplified example
     params.v_ptr = nullptr;  // Not used in this simplified example
     params.softmax_lse_ptr = nullptr;
     params.b = batch_size;
     params.h = num_heads;
     params.d = head_dim;
     params.seqlen_q = seqlen_q;
     params.seqlen_k = seqlen_k;
     params.total_q = seqlen_q;
     params.o_batch_stride = seqlen_q * num_heads * head_dim;
     params.o_row_stride = num_heads * head_dim;
     params.o_head_stride = head_dim;
     params.k_batch_stride = 0;
     params.k_row_stride = 0;
     params.k_head_stride = 0;
     params.v_batch_stride = 0;
     params.v_row_stride = 0;
     params.v_head_stride = 0;
     params.scale_softmax_log2 = 1.0f;
     params.softcap = 10.0f;
     params.unpadded_lse = false;
     params.seqlenq_ngroups_swapped = false;
     params.window_size_left = 0;
     params.window_size_right = 0;
     params.philox_args[0] = 0;
     params.philox_args[1] = 0;
     params.p_dropout_in_uint8_t = 0;
     params.h_h_k_ratio = 1;
     params.alibi_slopes_ptr = nullptr;
     params.alibi_slopes_batch_stride = 0;
 
     // Launch kernel
     flash_attention<KT, false, false, false, false, true, true, false, false>(params);
 
     // Copy results to host
     float h_output[batch_size * seqlen_q * num_heads * head_dim];
     cudaMemcpy(h_output, d_output, batch_size * seqlen_q * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
 
     // Print output
     printf("Output tensor:\n");
     for (int b = 0; b < batch_size; ++b) {
         printf("Batch %d:\n", b);
         for (int s = 0; s < seqlen_q; ++s) {
             for (int h = 0; h < num_heads; ++h) {
                 printf("Head %d, Seq %d: ", h, s);
                 for (int d = 0; d < head_dim; ++d) {
                     int idx = b * (seqlen_q * num_heads * head_dim) + s * (num_heads * head_dim) + h * head_dim + d;
                     printf("%f ", h_output[idx]);
                 }
                 printf("\n");
             }
         }
         printf("\n");
     }
 
     // Clean up
     cudaFree(d_output);
     return 0;
 }