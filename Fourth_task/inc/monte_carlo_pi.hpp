#pragma once
#include <fstream>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>

namespace pi_bench {
static const size_t ITERS = 100000000;
template <typename ElemType>
void runBenchmark(std::ofstream& output, size_t iterations, const std::string& typeName) {
    size_t inCircleUni  = 0;
    const ElemType ZERO = static_cast<ElemType>(0.0);
    const ElemType ONE = static_cast<ElemType>(1.0);
    const ElemType RADIUS = static_cast<ElemType>(1.0);
    const ElemType QUARTER_COEF = static_cast<ElemType>(4.0);

    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 genUni(rd() + static_cast<unsigned int>(omp_get_thread_num()));
        std::uniform_real_distribution<ElemType> distUni(ZERO, ONE);

        #pragma omp for reduction(+:inCircleUni)
        for (size_t i = 0; i < iterations; ++i) {
            ElemType x = distUni(genUni);
            ElemType y = distUni(genUni);
            if (x * x + y * y <= RADIUS) {
                ++inCircleUni;
            }
        }
    }

    ElemType calculatedPi = QUARTER_COEF * static_cast<ElemType>(inCircleUni) / static_cast<ElemType>(iterations);
    output << "[" << typeName << " on uniform_real_distribution" << "]\n";
    output << "  Calculated Pi: " << std::fixed << std::setprecision(15) << calculatedPi << "\n";
}
} // namespace pi_bench