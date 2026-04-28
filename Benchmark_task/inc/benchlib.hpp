#pragma once

//#include <immintrin.h>
#include <x86intrin.h>

#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

namespace benchlib
{
constexpr int ITERATIONS = 10000;
constexpr int GEN_AMOUNT = 10000;
constexpr int GEN_WARMUP = 1000;

template <typename Func, typename ElemType>
void funcLatencyTest(Func&& testFunc, const std::vector<ElemType>& data, std::ostream& output = std::cerr)
{
    using bitType          = std::conditional_t<sizeof(ElemType) == 8, uint64_t, uint32_t>;
    constexpr bitType ZERO = static_cast<bitType>(0ull);

    ElemType offset            = std::bit_cast<ElemType>(ZERO);
    volatile ElemType zeroMask = std::bit_cast<ElemType>(ZERO);
    auto size                  = data.size();
    std::vector<ElemType> res(size);
    uint64_t result = UINT64_MAX;

    for (size_t i = 0; i < size; ++i)
    {
        volatile auto dummy = testFunc(data[i]);
        static_cast<void>(dummy);
    }

    for (int n = 0; n != ITERATIONS; ++n)
    {
        _mm_lfence();
        uint64_t begin = __rdtsc();
        _mm_lfence();

        for (size_t i = 0; i != size; ++i)
        {
            res[i] = std::forward<Func>(testFunc)(data[i] + offset);

            bitType bits = std::bit_cast<bitType>(res[i]);
            bits &= std::bit_cast<bitType>(zeroMask);
            offset = std::bit_cast<ElemType>(bits);
        }

        _mm_lfence();
        uint64_t end = __rdtsc();
        _mm_lfence();
        result = std::min(result, end - begin);
    }

    double cpe = static_cast<double>(result) / size;
    output << "Latency: " << cpe << " CPE" << std::endl;
}

template <typename Func, typename ElemType>
void funcThroughputTest(Func&& testFunc, const std::vector<ElemType>& data, std::ostream& output = std::cerr)
{
    auto size = data.size();
    std::vector<ElemType> res(size);
    uint64_t result = UINT64_MAX;

    for (size_t i = 0; i < size; ++i)
    {
        volatile auto dummy = testFunc(data[i]);
        static_cast<void>(dummy);
    }

    for (int n = 0; n != ITERATIONS; ++n)
    {
        _mm_lfence();
        uint64_t begin = __rdtsc();
        _mm_lfence();

        for (size_t i = 0; i != size; ++i)
        {
            res[i] = std::forward<Func>(testFunc)(data[i]);
        }

        _mm_lfence();
        uint64_t end = __rdtsc();
        _mm_lfence();
        result = std::min(result, end - begin);
    }

    double cpe = static_cast<double>(result) / size;
    output << "Throughput: " << cpe << " CPE" << std::endl;
}

template <typename RNG>
void genLatencyTest(RNG& generator, std::ostream& output = std::cerr)
{
    uint64_t acc    = 0;
    uint64_t result = UINT64_MAX;

    for (int i = 0; i < GEN_WARMUP; ++i)
    {
        volatile auto dummy = generator();
        static_cast<void>(dummy);
    }

    for (int n = 0; n != ITERATIONS; ++n)
    {
        _mm_lfence();
        uint64_t begin = __rdtsc();
        _mm_lfence();

        for (int i = 0; i != GEN_AMOUNT; ++i)
        {
            acc ^= generator();
        }

        _mm_lfence();
        uint64_t end = __rdtsc();
        _mm_lfence();

        result = std::min(result, end - begin);
    }

    double cpe = static_cast<double>(result) / GEN_AMOUNT;
    output << "Latency: " << cpe << " CPE" << std::endl;
}

template <typename RNGFunc>
void genThroughputTest(std::ostream& output = std::cerr)
{
    RNGFunc rng1(1), rng2(2), rng3(3), rng4(4);
    uint64_t acc1, acc2, acc3, acc4;
    uint64_t result = UINT64_MAX;

    for (int i = 0; i < GEN_WARMUP; ++i)
    {
        volatile auto dummy1 = rng1();
        static_cast<void>(dummy1);

        volatile auto dummy2 = rng2();
        static_cast<void>(dummy2);

        volatile auto dummy3 = rng3();
        static_cast<void>(dummy3);

        volatile auto dummy4 = rng4();
        static_cast<void>(dummy4);
    }

    for (int n = 0; n != ITERATIONS; ++n)
    {
        _mm_lfence();
        uint64_t begin = __rdtsc();
        _mm_lfence();

        for (int i = 0; i != GEN_AMOUNT / 4; ++i)
        {
            acc1 ^= rng1();
            acc2 ^= rng2();
            acc3 ^= rng3();
            acc4 ^= rng4();
        }

        _mm_lfence();
        uint64_t end = __rdtsc();
        _mm_lfence();

        result = std::min(result, end - begin);
    }

    double cpe = static_cast<double>(result) / (GEN_AMOUNT * 4);
    output << "Throughput: " << cpe << " CPE" << std::endl;
}

}  // namespace benchlib