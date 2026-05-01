#include <immintrin.h>
#include <omp.h>
#include <sleef.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <vector>

#include "loglib.hpp"
#include "minstdrand.hpp"
#include "option_lib.hpp"

namespace financial
{
void calcSimdBS(const OptionParameters& opt, uint64_t paths, const uint64_t& steps, double& outCall, double& outPut)
{
    float dt      = static_cast<float>(opt.timeToMaturity_) / steps;
    float sqrt_dt = std::sqrt(dt);

    // 1024 float = 4KB. 4 буфера = 16KB, отлично помещается в L1d-кэш
    constexpr uint64_t BLOCK = 1024;

    double totalPayoffCall = 0.0;
    double totalPayoffPut  = 0.0;

#pragma omp parallel reduction(+ : totalPayoffCall, totalPayoffPut)
    {
        rng::VectorMinstd<8> rng;
        uint64_t pathPerThreads = paths / static_cast<uint64_t>(omp_get_num_threads());
        uint64_t threadOffset   = static_cast<uint64_t>(omp_get_thread_num()) * pathPerThreads * steps * 2;
        rng.seed(42, threadOffset);

        // выравнивание для эффективного SIMD Load/Store
        alignas(32) float curPriceBlock[BLOCK];
        alignas(32) float u1Block[BLOCK];
        alignas(32) float u2Block[BLOCK];
        alignas(32) float lnCalcResBlock[BLOCK];

        const __m256 v_drift     = _mm256_set1_ps(1.0f + opt.riskFreeRate_ * dt);
        const __m256 v_vol       = _mm256_set1_ps(opt.volatility_ * sqrt_dt);
        const __m256 v_K         = _mm256_set1_ps(opt.strikePrice_);
        const __m256 v_zero      = _mm256_setzero_ps();
        const __m256 v_minus_two = _mm256_set1_ps(-2.0f);
        const __m256 v_2pi       = _mm256_set1_ps(6.28318530718f);

#pragma omp for schedule(static)
        for (uint64_t p = 0; p < paths; p += BLOCK)
        {
            for (uint64_t i = 0; i < BLOCK; ++i) curPriceBlock[i] = opt.spotPrice_;

            for (uint64_t t = 0; t < steps; ++t)  // интегрирование по времени
            {
                rng.generateFloat(u1Block);
                rng.generateFloat(u2Block);

// меняем диапазон [-1, 1] -> (0, 1] для вычисления логарифма
#pragma omp simd
                for (uint64_t i = 0; i < BLOCK; ++i)
                {
                    u1Block[i] = std::max(u1Block[i] * 0.5f + 0.5f, 1e-10f);
                    u2Block[i] = u2Block[i] * 0.5f + 0.5f;
                }

                math::logf_avx(u1Block, lnCalcResBlock, BLOCK);

                for (size_t j = 0; j < BLOCK; j += 8)
                {
                    __m256 ln_u1 = _mm256_load_ps(&lnCalcResBlock[j]);
                    __m256 u2    = _mm256_load_ps(&u2Block[j]);

                    // ПОЛНЫЙ Бокс-Мюллер
                    __m256 r     = _mm256_sqrt_ps(_mm256_mul_ps(v_minus_two, ln_u1));
                    __m256 theta = _mm256_mul_ps(v_2pi, u2);
                    __m256 z     = _mm256_mul_ps(r, Sleef_cosf8_u10(theta));

                    __m256 s      = _mm256_load_ps(&curPriceBlock[j]);
                    __m256 factor = _mm256_fmadd_ps(v_vol, z, v_drift);
                    s             = _mm256_mul_ps(s, factor);
                    _mm256_store_ps(&curPriceBlock[j], s);
                }
            }

            // вычисление выплаты в конце
            for (uint64_t j = 0; j < BLOCK; j += 8)
            {
                __m256 s = _mm256_load_ps(&curPriceBlock[j]);

                // Call = max(S - K, 0), Put = max(K - S, 0)
                __m256 payoff_c = _mm256_max_ps(_mm256_sub_ps(s, v_K), v_zero);
                __m256 payoff_p = _mm256_max_ps(_mm256_sub_ps(v_K, s), v_zero);

                alignas(32) float c_res[8], p_res[8];
                _mm256_store_ps(c_res, payoff_c);
                _mm256_store_ps(p_res, payoff_p);

                for (int k = 0; k < 8; ++k)
                {
                    totalPayoffCall += c_res[k];
                    totalPayoffPut += p_res[k];
                }
            }
        }
    }

    double discount = std::exp(-opt.riskFreeRate_ * opt.timeToMaturity_);
    outCall         = discount * (totalPayoffCall / paths);
    outPut          = discount * (totalPayoffPut / paths);
}

}  // namespace financial