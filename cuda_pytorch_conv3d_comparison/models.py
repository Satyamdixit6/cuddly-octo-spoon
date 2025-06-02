import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the custom CUDA operator
# This assumes you've compiled it and it's in yourPYTHONPATH
# or installed in your environment.
# The name 'conv3d_custom_cuda_op._C' comes from setup.py
try:
    import conv3d_custom_cuda_op._C as custom_conv3d_op
    CUDA_EXT_AVAILABLE = True
except ImportError:
    print("Warning: Custom CUDA 3D convolution operator not found. Build it first.")
    CUDA_EXT_AVAILABLE = False


class CustomConv3dCUDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv3dCUDA, self).__init__()
        if not CUDA_EXT_AVAILABLE:
            raise RuntimeError("Custom CUDA 3D Conv extension not compiled or not found.")

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        assert len(self.kernel_size) == 3, "kernel_size must be an int or a tuple of 3 ints"

        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride
        assert len(self.stride) == 3, "stride must be an int or a tuple of 3 ints"

        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding
        assert len(self.padding) == 3, "padding must be an int or a tuple of 3 ints"

        # K, C, D_f, H_f, W_f
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # Or other init
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: N, C, D_in, H_in, W_in
        # weight: K, C, D_f, H_f, W_f
        # bias: K
        return custom_conv3d_op.forward(
            x, self.weight, self.bias if self.bias is not None else torch.empty(0, device=x.device), # Pass empty tensor if no bias
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2]
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
                f"bias={self.bias is not None})")


class PyTorchConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(PyTorchConv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):
        return self.conv(x)

    def get_weights(self):
        return self.conv.weight, self.conv.bias

    def set_weights(self, weight, bias):
        self.conv.weight = nn.Parameter(weight.clone())
        if bias is not None and self.conv.bias is not None:
            self.conv.bias = nn.Parameter(bias.clone())
        elif bias is None and self.conv.bias is not None:
             self.conv.bias = None # Or handle appropriately
        elif bias is not None and self.conv.bias is None:
            # This case needs careful handling if PyTorch layer was init with bias=False
            # For simplicity, we assume if one has bias, the other can too.
             if self.conv.bias is None and bias is not None: # Create bias if PyTorch layer didn't have one
                self.conv.bias = nn.Parameter(bias.clone())


    def __repr__(self):
        return (f"{self.__class__.__name__}({self.conv.in_channels}, {self.conv.out_channels}, "
                f"kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding}, "
                f"bias={self.conv.bias is not None})")