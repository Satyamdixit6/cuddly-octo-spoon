import torch
import my_custom_ops_cuda # Name from setup.py

# Ensure CUDA is available before running
if torch.cuda.is_available():
    a_gpu = torch.randn(1024, device='cuda', dtype=torch.float32)
    b_gpu = torch.randn(1024, device='cuda', dtype=torch.float32)

    c_gpu = my_custom_ops_cuda.vector_add(a_gpu, b_gpu)

    c_pytorch = a_gpu + b_gpu
    print(f"Custom CUDA op matches PyTorch op: {torch.allclose(c_gpu, c_pytorch)}")
    print(c_gpu)
else:
    print("CUDA device not found. Skipping PyTorch extension test.")