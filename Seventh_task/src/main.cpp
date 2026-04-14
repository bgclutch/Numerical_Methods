#include "matrix.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <x86intrin.h>

template <typename Func>
void measureCycles(const std::string& name, Func&& func, size_t ops) {
    func();
    unsigned long long start = __rdtsc();
    func();
    unsigned long long end = __rdtsc();
    unsigned long long cycles = end - start;
    double flops_per_cycle = static_cast<double>(ops) / cycles;

    std::cout << "[*] " << name << " took: " << cycles << " cycles\n";
    std::cout << "    Performance: " << flops_per_cycle << " FLOPS/cycle\n";
}

int main() {
    try {
        std::cout << "Initializing MatrixSet (fp32)...\n";
        matrix::MatrixSet<float> matSet;

        size_t M = matSet.getRows_1();
        size_t K = matSet.getCols_1();
        size_t N = matSet.getCols_2();

        std::cout << "Matrix A: " << M << "x" << K << " | Matrix B: " << K << "x" << N << "\n";

        size_t total_flops = 2 * M * N * K;

        std::cout << "Total Floating-Point Ops (FLOPS): " << total_flops << "\n";
        std::cout << "--------------------------------------------------\n";

        // 1. naive multiplication
        measureCycles("1. Naive Mult", [&]() {
            matSet.naiveMult();
        }, total_flops);
        std::vector<float> naiveResult = matSet.getResult();

        // 2. vectorized multiplication
        measureCycles("2. Auto-vectorized", [&]() {
            matSet.vectMult();
        }, total_flops);

        // 3. AVX multiplication
        measureCycles("3. AVX Intrinsic Mult", [&]() {
            matSet.intrinsicMult();
        }, total_flops);

        // 3.1. AVX improved multiplication
        measureCycles("3.1. AVX improved Intrinsic Mult", [&]() {
            matSet.intrinsicMultImproved();
        }, total_flops);

        // 3.2. AVX improved && tiled multiplication
        measureCycles("3.2. AVX improved && Tiled Intrinsic Mult", [&]() {
            matSet.intrinsicMultTiled();
        }, total_flops);

        // 3.2. AVX improved && tiled multiplication
        measureCycles("3.3. AVX twice improved && Tiled Intrinsic Mult", [&]() {
            matSet.intrinsicMultAbsolute();
        }, total_flops);

        // 4. GPU multiplication
        measureCycles("4. OpenCL GPU Mult", [&]() {
            matSet.GPUMult();
        }, total_flops);
        std::vector<float> gpuResult = matSet.getResult();

        std::cout << "--------------------------------------------------\n";

        bool isCorrect = true;
        for (size_t i = 0; i != naiveResult.size(); ++i) {
            float diff = std::abs(naiveResult[i] - gpuResult[i]);
            float ref  = std::abs(naiveResult[i]);

            if (diff > 0.005f * ref && diff > 0.5f) {
                isCorrect = false;
                break;
            }
        }

        if (isCorrect)
            std::cout << "SUCCESS: Functional correctness verified (Naive == GPU)!\n";
        else
            std::cout << "WARNING: Results mismatch!\n";

    } catch (const std::exception& e) {
        std::cerr << "\nERROR" << e.what() << "\n";
    } catch (...) {
        std::cerr << "\nUNEXPECTED ERROR\n";
    }


    return 0;
}