#include <immintrin.h>
#include <omp.h>
#include <sleef.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <span>
#include <vector>

#include "loglib.hpp"
#include "minstdrand.hpp"
#include "option_lib.hpp"

namespace financial
{
void calcOptionsMC(const std::vector<OptionParameters>& portfolio, const uint64_t& pathPerOption, std::vector<MCResult>& results)
{
    const uint64_t numOptions = portfolio.size();
    results.resize(numOptions);

#pragma omp parallel for schedule(dynamic)
    for (uint64_t i = 0; i < numOptions; ++i)
    {
        const auto& opt = portfolio[i];

        uint64_t offset = static_cast<uint64_t>(i) * pathPerOption * 2ULL;
        rng::VectorMinstd<8> rng;
        rng.seed(42, offset);

        double T = opt.timeToMaturity_, r = opt.riskFreeRate_, v = opt.volatility_, S0 = opt.spotPrice_, K = opt.strikePrice_;

        float drift    = (float)((r - 0.5 * v * v) * T);
        float volSqrtT = (float)(v * std::sqrt(T));

        const __m256 v_drift = _mm256_set1_ps(drift);
        const __m256 v_vol   = _mm256_set1_ps(volSqrtT);
        const __m256 v_S0    = _mm256_set1_ps((float)S0);
        const __m256 v_K     = _mm256_set1_ps((float)K);
        const __m256 v_zero  = _mm256_setzero_ps();
        const __m256 v_2pi   = _mm256_set1_ps(6.283185307f);
        const __m256 v_m2    = _mm256_set1_ps(-2.0f);
        const __m256 v_05    = _mm256_set1_ps(0.5f);

        __m256d total_l = _mm256_setzero_pd();
        __m256d total_h = _mm256_setzero_pd();

        constexpr uint64_t BLOCK = 1024;
        alignas(32) float u1[BLOCK], u2[BLOCK], ln1[BLOCK];

        // будем генерировать pathPerOption / 2 пар и из каждой делать 4 пути
        // для удобства оставим цикл по pathPerOption и просто удвоим выборку
        for (uint64_t p = 0; p < pathPerOption; p += BLOCK)
        {
            uint64_t cur_size = std::min(BLOCK, pathPerOption - p);
            rng.generateFloat(u1);
            rng.generateFloat(u2);

            for (uint64_t k = 0; k < BLOCK; k += 8)
            {
                __m256 v1 = _mm256_load_ps(&u1[k]);
                __m256 v2 = _mm256_load_ps(&u2[k]);
                // из генератора получаем [-1, 1] -> [0, 1] для расчета логарифма
                v1 = _mm256_fmadd_ps(v1, v_05, v_05);
                v1 = _mm256_max_ps(v1, _mm256_set1_ps(1e-12f));
                v2 = _mm256_fmadd_ps(v2, v_05, v_05);
                _mm256_store_ps(&u1[k], v1);
                _mm256_store_ps(&u2[k], v2);
            }

            math::logf_avx(u1, ln1, BLOCK);

            for (uint64_t j = 0; j < cur_size; j += 8)
            {
                __m256 ln_v1 = _mm256_load_ps(&ln1[j]);
                __m256 v2    = _mm256_load_ps(&u2[j]);

                __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(v_m2, ln_v1));
                __m256 angle  = _mm256_mul_ps(v_2pi, v2);

                // расчет Z0 через cos и Z1 через sin
                __m256 z0 = _mm256_mul_ps(radius, Sleef_cosf8_u10(angle));
                __m256 z1 = _mm256_mul_ps(radius, Sleef_sinf8_u10(angle));

                // для каждого Z считаем +Z и -Z
                auto calcPayoff = [&](__m256 z)
                {
                    __m256 st_up   = _mm256_mul_ps(v_S0, Sleef_expf8_u10(_mm256_fmadd_ps(v_vol, z, v_drift)));
                    __m256 st_down = _mm256_mul_ps(v_S0, Sleef_expf8_u10(_mm256_fnmadd_ps(v_vol, z, v_drift)));
                    return _mm256_mul_ps(
                        _mm256_add_ps(_mm256_max_ps(_mm256_sub_ps(st_up, v_K), v_zero), _mm256_max_ps(_mm256_sub_ps(st_down, v_K), v_zero)),
                        v_05);
                };

                __m256 payoff0      = calcPayoff(z0);
                __m256 payoff1      = calcPayoff(z1);
                __m256 payoff_final = _mm256_mul_ps(_mm256_add_ps(payoff0, payoff1), v_05);  // среднее по всем путям

                total_l = _mm256_add_pd(total_l, _mm256_cvtps_pd(_mm256_extractf128_ps(payoff_final, 0)));
                total_h = _mm256_add_pd(total_h, _mm256_cvtps_pd(_mm256_extractf128_ps(payoff_final, 1)));
            }
        }

        alignas(32) double res[8];
        _mm256_store_pd(&res[0], total_l);
        _mm256_store_pd(&res[4], total_h);

        double final_sum = 0;
        for (int k = 0; k < 8; ++k) final_sum += res[k];

        results[i].callPrice = std::exp(-r * T) * (final_sum / (double)pathPerOption);
    }
}
}  // namespace financial
