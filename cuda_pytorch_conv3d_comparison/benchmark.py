import torch
import time
import numpy as np
from models import CustomConv3dCUDA, PyTorchConv3d, CUDA_EXT_AVAILABLE

def get_output_shape(input_shape, kernel_size, stride, padding):
    N, C_in, D_in, H_in, W_in = input_shape
    KD, KH, KW = kernel_size
    SD, SH, SW = stride
    PD, PH, PW = padding

    D_out = (D_in + 2 * PD - KD) // SD + 1
    H_out = (H_in + 2 * PH - KH) // SH + 1
    W_out = (W_in + 2 * PW - KW) // SW + 1
    return N, D_out, H_out, W_out # C_out is determined by layer's out_channels

def benchmark_layer(layer_class, layer_name, input_tensor,
                    in_channels, out_channels, kernel_size, stride, padding, bias,
                    num_warmup=50, num_runs=200):
    print(f"\n--- Benchmarking {layer_name} ---")
    if layer_name == "Custom CUDA Conv3D" and not CUDA_EXT_AVAILABLE:
        print("Custom CUDA extension not available. Skipping.")
        return {
            "name": layer_name,
            "avg_latency_ms": float('nan'),
            "std_latency_ms": float('nan'),
            "throughput_samples_sec": float('nan'),
            "peak_memory_allocated_mb": float('nan'),
            "output_correct": "N/A (Skipped)"
        }

    device = input_tensor.device
    layer = layer_class(in_channels, out_channels, kernel_size, stride, padding, bias=bias).to(device)
    layer.eval() # Set to evaluation mode

    # For custom layer, ensure its weights are on the correct device
    if isinstance(layer, CustomConv3dCUDA):
        layer.weight.data = layer.weight.data.to(device)
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.to(device)

    # Ensure PyTorch layer also has its parameters on the correct device (usually automatic)
    if isinstance(layer, PyTorchConv3d):
         pass # nn.Conv3d handles device placement with .to(device)

    # --- Correctness Check (Optional, but good for debugging) ---
    # We can compare with PyTorch's Conv3d if this IS the custom layer
    # For this, we need to ensure weights are the same.
    output_correct_str = "N/A"
    if layer_name == "Custom CUDA Conv3D":
        try:
            pytorch_ref_layer = PyTorchConv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias).to(device)
            pytorch_ref_layer.set_weights(layer.weight.data.clone(), layer.bias.data.clone() if layer.bias is not None else None)
            pytorch_ref_layer.eval()

            with torch.no_grad():
                custom_output = layer(input_tensor)
                pytorch_output = pytorch_ref_layer(input_tensor)

            if torch.allclose(custom_output, pytorch_output, atol=1e-4, rtol=1e-3): # Looser tolerance for FP32
                output_correct_str = "Matches PyTorch"
            else:
                diff = torch.abs(custom_output - pytorch_output)
                output_correct_str = f"MISMATCH vs PyTorch! Max diff: {diff.max().item():.4e}, Mean diff: {diff.mean().item():.4e}"
                # print("Custom output sample:", custom_output.flatten()[:5])
                # print("PyTorch output sample:", pytorch_output.flatten()[:5])
        except Exception as e:
            output_correct_str = f"Correctness check failed: {e}"
    elif layer_name == "PyTorch nn.Conv3d":
        output_correct_str = "Reference"


    print(f"[{layer_name}] Weights initialized on: {layer.weight.device if hasattr(layer, 'weight') else layer.conv.weight.device}")
    print(f"[{layer_name}] Input tensor on: {input_tensor.device}")


    # Warmup runs
    for _ in range(num_warmup):
        _ = layer(input_tensor)
    torch.cuda.synchronize()

    # Latency and Throughput
    latencies = []
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device)

    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = layer(input_tensor)
        end_event.record()
        torch.cuda.synchronize() # Wait for the operation to complete

        latencies.append(start_event.elapsed_time(end_event)) # Time in ms

    peak_memory_allocated = torch.cuda.max_memory_allocated(device) - initial_memory # More accurate delta

    avg_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)
    # Throughput: samples (batch size) processed per second
    throughput = (input_tensor.size(0) / (avg_latency_ms / 1000.0)) if avg_latency_ms > 0 else 0

    print(f"Output Correctness: {output_correct_str}")
    print(f"Average Latency: {avg_latency_ms:.3f} ms")
    print(f"Std Dev Latency: {std_latency_ms:.3f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Peak Memory Allocated (Delta): {peak_memory_allocated / (1024**2):.2f} MB")

    return {
        "name": layer_name,
        "avg_latency_ms": avg_latency_ms,
        "std_latency_ms": std_latency_ms,
        "throughput_samples_sec": throughput,
        "peak_memory_allocated_mb": peak_memory_allocated / (1024**2),
        "output_correct": output_correct_str
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        exit()

    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    # --- Configuration ---
    batch_size = 8
    in_channels_list = [3] # [3, 16, 32]
    out_channels_list = [16] # [16, 32, 64]
    input_spatial_depth = [16] # [8, 16, 32] # D_in
    input_spatial_size = [64] # [32, 64, 128] # H_in, W_in (assuming square)
    kernel_sizes = [(3,3,3)] # [(3,3,3), (5,5,5)]
    strides = [(1,1,1)] # [(1,1,1), (2,2,2)]
    paddings = [(1,1,1)] # [(0,0,0), (1,1,1)] # 'same' padding for stride 1, kernel 3 is padding 1
    biases = [True, False]

    results_summary = []

    for C_in in in_channels_list:
        for K_out in out_channels_list:
            for D_s in input_spatial_depth:
                for S_s in input_spatial_size: # H_in = W_in = S_s
                    for kernel_s in kernel_sizes:
                        for stride_s in strides:
                            for padding_s in paddings:
                                for bias_val in biases:
                                    input_shape = (batch_size, C_in, D_s, S_s, S_s)
                                    
                                    # Ensure padding is not too large for kernel/stride
                                    valid_padding = True
                                    for i in range(3):
                                        if padding_s[i] > kernel_s[i] // 2 and stride_s[i] == 1: # Basic check
                                             pass # Allow 'same' style padding
                                        # More complex checks might be needed for arbitrary padding

                                    if not valid_padding:
                                        print(f"Skipping invalid padding {padding_s} for kernel {kernel_s}")
                                        continue
                                    
                                    # Check if output dimensions will be positive
                                    _, D_out, H_out, W_out = get_output_shape(input_shape, kernel_s, stride_s, padding_s)
                                    if D_out <=0 or H_out <=0 or W_out <=0:
                                        print(f"Skipping config due to non-positive output dim: In={input_shape} K={kernel_s} S={stride_s} P={padding_s} -> Out=({D_out},{H_out},{W_out})")
                                        continue


                                    print(f"\n======================================================================")
                                    print(f"Benchmarking Config: Input={input_shape}, OutChannels={K_out}, "
                                          f"Kernel={kernel_s}, Stride={stride_s}, Padding={padding_s}, Bias={bias_val}")
                                    print(f"======================================================================")

                                    # Generate a random input tensor
                                    input_tensor = torch.randn(input_shape, device=device, dtype=torch.float32)

                                    # Benchmark PyTorch's nn.Conv3d
                                    pytorch_results = benchmark_layer(
                                        PyTorchConv3d, "PyTorch nn.Conv3d", input_tensor,
                                        C_in, K_out, kernel_s, stride_s, padding_s, bias_val
                                    )
                                    results_summary.append({**pytorch_results, "config": f"In{input_shape}-K{K_out}-ker{kernel_s}-S{stride_s}-P{padding_s}-B{bias_val}"})


                                    # Benchmark Custom CUDA Conv3D
                                    if CUDA_EXT_AVAILABLE:
                                        custom_results = benchmark_layer(
                                            CustomConv3dCUDA, "Custom CUDA Conv3D", input_tensor,
                                            C_in, K_out, kernel_s, stride_s, padding_s, bias_val
                                        )
                                        results_summary.append({**custom_results, "config": f"In{input_shape}-K{K_out}-ker{kernel_s}-S{stride_s}-P{padding_s}-B{bias_val}"})
                                    else:
                                        print("\nSkipping Custom CUDA Conv3D benchmark as extension is not available.")
                                        results_summary.append({
                                            "name": "Custom CUDA Conv3D",
                                            "avg_latency_ms": float('nan'), "std_latency_ms": float('nan'),
                                            "throughput_samples_sec": float('nan'), "peak_memory_allocated_mb": float('nan'),
                                            "output_correct": "N/A (Skipped)",
                                            "config": f"In{input_shape}-K{K_out}-ker{kernel_s}-S{stride_s}-P{padding_s}-B{bias_val}"
                                        })


    print("\n\n--- Benchmark Summary ---")
    print(f"{'Configuration':<70} | {'Layer':<20} | {'Latency (ms)':<15} | {'Throughput (samp/s)':<20} | {'Memory (MB)':<12} | {'Correctness':<20}")
    print("-" * 160)
    for res in results_summary:
        print(f"{res['config']:<70} | {res['name']:<20} | {res['avg_latency_ms']:.3f}{('' if np.isnan(res['avg_latency_ms']) else ' +/- ' + f'{res["std_latency_ms"]:.3f}'):<15} | "
              f"{res['throughput_samples_sec']:.2f}{'':<20} | {res['peak_memory_allocated_mb']:.2f}{'':<12} | {res['output_correct']:<20}")