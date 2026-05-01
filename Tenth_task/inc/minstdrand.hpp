#pragma once

#include <immintrin.h>

#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace rng
{
namespace detail
{
template <typename ElemType = uint64_t>
ElemType modPow(ElemType base, ElemType exp, ElemType mod)
{
    ElemType res = 1;
    base %= mod;
    while (exp > 0)
    {
        if (exp % 2 == 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return res;
}
}  // namespace detail

template <uint64_t Dimension = 8>
class VectorMinstd
{
private:
    static constexpr uint64_t a      = 48271;
    static constexpr uint64_t period = 2147483647;  // == 2^31 - 1
    uint32_t Ak32;
    std::vector<uint32_t> state;

public:
    VectorMinstd() : state(Dimension)
    {
        uint64_t Ak = detail::modPow(a, Dimension, period);
        Ak32        = static_cast<uint32_t>(Ak);
    }

    void seed(uint32_t seed, uint64_t skip = 0)
    {
        uint64_t current = seed;
        if (skip > 0)
        {
            // пропускаем skip шагов
            current = (current * detail::modPow(a, skip, period)) % period;
        }

        // в C++ std::minstd_rand отдает значения из состояния после seed
        current  = (current * a) % period;
        state[0] = static_cast<uint32_t>(current);

        for (size_t i = 1; i < Dimension; ++i)
        {
            current  = (current * a) % period;
            state[i] = static_cast<uint32_t>(current);
        }
    }

    void generateFloat(std::span<float> out)
    {
        size_t size = out.size();
        for (size_t i = 0; i < size; i += Dimension)
        {
#pragma omp simd
            for (size_t j = 0; j < Dimension; ++j)
            {
                uint32_t current = state[j];
                out[i + j]       = static_cast<float>(current) * (2.0f / 2147483647.0f) - 1.0f;
                uint64_t y       = static_cast<uint64_t>(Ak32) * current;

                y = (y >> 31) + (y & 0x7FFFFFFF);  // тк число 2^31 - 1 простое можно найти отстаток от деления таким образом
                y = (y >> 31) + (y & 0x7FFFFFFF);

                state[j] = static_cast<uint32_t>(y);
            }
        }
    }

    void generateInt(std::span<uint32_t> out)
    {
        size_t size = out.size();
        for (size_t i = 0; i < size; i += Dimension)
        {
#pragma omp simd
            for (size_t j = 0; j < Dimension; ++j)
            {
                uint32_t current = state[j];
                out[i + j]       = current;

                uint64_t y = static_cast<uint64_t>(Ak32) * current;
                y          = (y >> 31) + (y & 0x7FFFFFFF);  // аналогично методу generateUniform
                y          = (y >> 31) + (y & 0x7FFFFFFF);

                state[j] = static_cast<uint32_t>(y);
            }
        }
    }
};
}  // namespace rng