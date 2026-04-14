#pragma once
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"

namespace benchmark
{
struct BenchTimes
{
    std::chrono::nanoseconds kernelTime{0};
    std::chrono::nanoseconds WallTime{0};
    std::chrono::nanoseconds TransferTime{0};
    std::chrono::nanoseconds CPUTime{0};
};
}  // namespace benchmark

namespace ocl_utils
{
class Environment final
{
private:
    cl::Platform platform_;
    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Program program_;
    cl::Kernel kernel_;
    cl::string kernel_name_;

public:
    Environment(const std::string& kernel_path, const std::string& kernel_name)
    {
        platform_    = select_platform();
        device_      = select_device(platform_);
        context_     = create_context(device_);
        queue_       = create_queue(context_, device_);
        program_     = create_program(context_, device_, kernel_path);
        kernel_      = create_kernel(program_, kernel_name);
        kernel_name_ = select_kernel_name(kernel_name);
    };

    Environment(const Environment& other, const std::string& kernel_path, const std::string& kernel_name)
        : platform_(other.platform_), device_(other.device_), context_(other.context_), queue_(other.queue_)
    {
        program_     = create_program(context_, device_, kernel_path);
        kernel_      = create_kernel(program_, kernel_name);
        kernel_name_ = select_kernel_name(kernel_name);
    };

    cl::Device& get_device() noexcept { return device_; }

    cl::Platform& get_platform() noexcept { return platform_; }

    cl::Context& get_context() noexcept { return context_; }

    cl::Program& get_program() noexcept { return program_; }

    cl::CommandQueue& get_queue() noexcept { return queue_; }

    cl::Kernel& get_kernel() noexcept { return kernel_; }

    cl::string& get_kernel_name() noexcept { return kernel_name_; }

private:
    cl::Platform select_platform()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty())
        {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        cl::Platform best_platform = platforms.front();
        long long max_score        = -1;

        for (const auto& plt : platforms)
        {
            std::vector<cl::Device> devices;
            plt.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (const auto& dev : devices)
            {
                long long current_score = 0;

                cl_device_type type = dev.getInfo<CL_DEVICE_TYPE>();
                cl_uint units       = dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

                if (type & CL_DEVICE_TYPE_GPU)
                {
                    current_score += 100000;

                    std::string vendor = dev.getInfo<CL_DEVICE_VENDOR>();
                    if (vendor.find("Intel") == std::string::npos) current_score += 50000;
                }
                else if (type & CL_DEVICE_TYPE_CPU)
                {
                    current_score += 1000;
                }

                current_score += units;

                if (current_score > max_score)
                {
                    max_score     = current_score;
                    best_platform = plt;
                }
            }
        }

        if (max_score == -1)
        {
            throw std::runtime_error("No valid OpenCL devices found in any platform.");
        }
        #if 0
        std::cerr << "Selected Platform: " << best_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        #endif
        return best_platform;
    }

    cl::Device select_device(cl::Platform& platform)
    {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if (devices.empty()) throw std::runtime_error("No devices found on selected platform");

        return devices.front();
    }

    cl::Program create_program(cl::Context& context, cl::Device& device, const std::string& kernel_path)
    {
        std::ifstream bitonicKernelFile(kernel_path);

        if (!bitonicKernelFile.is_open())
        {
            std::cerr << "CRITICAL ERROR: Could not open kernel file!" << std::endl;
            std::cerr << "Looked at: " << kernel_path << std::endl;
            throw std::runtime_error("File not found");
        }

        std::string sourceCode((std::istreambuf_iterator<char>(bitonicKernelFile)), std::istreambuf_iterator<char>());
        cl::Program::Sources sources;
        sources.push_back({sourceCode.c_str(), sourceCode.length()});

        if (sourceCode.empty())
        {
            std::cerr << "CRITICAL ERROR: Kernel file is EMPTY!" << std::endl;
            throw std::runtime_error("Empty source code");
        }

        cl::Program program(context, sources);
        if (program.build({device}) != CL_SUCCESS)
        {
            std::cerr << "Build Log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            throw std::runtime_error("program wasn't built");
        }

        return program;
    }

    cl::Context create_context(cl::Device& device)
    {
        cl::Context ret_obj(device);
        return ret_obj;
    }

    cl::CommandQueue create_queue(cl::Context& context, cl::Device& device)
    {
        cl::CommandQueue ret_obj(context, device);
        return ret_obj;
    }

    cl::Kernel create_kernel(cl::Program& program, const std::string& kernel_name)
    {
        cl_int err;
        cl::Kernel kernel(program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS)
        {
            std::cerr << "GPU Error: " << err << std::endl;
            throw std::runtime_error("no kernel creation");
        }

        if (err == CL_INVALID_KERNEL_NAME)
        {
            std::cerr << "Check spelling in .cl file!" << std::endl;
        }
        return kernel;
    }

    cl::string select_kernel_name(const std::string& kernel_name) { return kernel_name; }
};
}  // namespace ocl_utils