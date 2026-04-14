#pragma once
#include <string>

namespace config
{
#ifdef KERNELS_ABS_PATH
const std::string KERNELS_PATH = KERNELS_ABS_PATH;
#else
const std::string KERNELS_PATH = "";
#endif

const std::string MATMUL_KERNEL      = "naive_kernel.cl";
const std::string MATMUL_KERNEL_NAME = "naive_bitonic_sort_kernel";
}  // namespace config