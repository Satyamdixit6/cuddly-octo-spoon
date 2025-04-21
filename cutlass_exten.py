import torch

if torch.cuda.is_available():
    device = 'cuda'
    M, N, K = 512, 1024, 256
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)

    # torch.matmul uses highly optimized backend libraries (like cuBLAS)
    C = torch.matmul(A, B)
    print(f"PyTorch matmul output shape: {C.shape}")
else:
    print("CUDA not available for PyTorch MatMul example.")