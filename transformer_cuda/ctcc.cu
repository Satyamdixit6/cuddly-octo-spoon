#include<cuda_runtime.h>
#include<torch/tensor.h>
#include<iostream.h>

// first i will build the simple attention and ffn

__global__ void  attentin_(const float * Q , const float*K , const float*V , int seq_len, int head_dim)