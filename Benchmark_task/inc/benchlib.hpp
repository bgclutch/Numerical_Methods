#pragma once

#include <immintrin.h>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>

namespace benchlib
{
constexpr int ITERATIONS = 10000;
constexpr int GEN_AMOUNT = 10000;
constexpr int GEN_WARMUP = 1000;

struct ResData
{
    double mean;
    double stddev;
};

namespace detail
{
template <typename ElemType = double>
ResData calculateStats(const std::vector<ElemType>& data)
{
    double sum = 0;

    for (auto x : data) sum += x;

    size_t size = data.size();
    double mean = sum / size;

    double varianceSum = 0;

    for (auto x : data)
    {
        double diff = x - mean;
        varianceSum += diff * diff;
    }

    double stddev = std::sqrt(varianceSum / (size - 1));

    return ResData{mean, stddev};
}
}  // namespace detail

template <typename Func, typename ElemType>
void funcLatencyTest(Func&& testFunc, const std::vector<ElemType>& data, std::ostream& output = std::cerr)
{
    using bitType          = std::conditional_t<sizeof(ElemType) == 8, uint64_t, uint32_t>;
    constexpr bitType ZERO = static_cast<bitType>(0ull);

    ElemType offset            = std::bit_cast<ElemType>(ZERO);
    volatile ElemType zeroMask = std::bit_cast<ElemType>(ZERO);
    auto size                  = data.size();
    std::vector<ElemType> res(size);
    std::vector<double> dataVec;

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

        uint32_t aux;
        uint64_t end = __rdtscp(&aux);
        _mm_lfence();
        dataVec.push_back((end - begin) / GEN_AMOUNT);
    }

    ResData result = detail::calculateStats(dataVec);
    output << "Latency: " << result.mean << " CPE +-" << result.stddev << std::endl;
}

template <typename Func, typename ElemType>
void funcThroughputTest(Func&& testFunc, const std::vector<ElemType>& data, std::ostream& output = std::cerr)
{
    auto size = data.size();
    std::vector<ElemType> res(size);
    std::vector<double> dataVec;

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

        uint32_t aux;
        uint64_t end = __rdtscp(&aux);
        _mm_lfence();

        dataVec.push_back((end - begin) / GEN_AMOUNT);
    }

    ResData result = detail::calculateStats(dataVec);
    output << "Throughput: " << result.mean << " CPE +-" << result.stddev << std::endl;
}

template <typename Func, typename ElemType>
void vfuncTest(Func&& testFunc, const std::vector<ElemType>& data, std::ostream& output = std::cerr) {
    size_t size = data.size();
    std::vector<ElemType> res(size);
    std::vector<double> dataVec;

    for (int n = 0; n != ITERATIONS; ++n) {
        _mm_lfence();
        uint64_t begin = __rdtsc();
        _mm_lfence();

        std::forward<Func>(testFunc)(data, res, size);

        unsigned int aux;
        uint64_t end = __rdtscp(&aux);
        _mm_lfence();

        dataVec.push_back(static_cast<double>(end - begin) / size);

        volatile ElemType dummy = res[size - 1];
        static_cast<void>(dummy);
    }

    ResData result = detail::calculateStats(dataVec);
    output << "Throughput: " << result.mean << " CPE +-" << result.stddev << std::endl;
}

template <typename RNG>
void genLatencyTest(RNG& generator, std::ostream& output = std::cerr)
{
    uint64_t acc = 0;
    std::vector<double> dataVec;

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

        uint32_t aux;
        uint64_t end = __rdtscp(&aux);
        _mm_lfence();

        dataVec.push_back((end - begin) / GEN_AMOUNT);
    }

    ResData result = detail::calculateStats(dataVec);
    output << "Latency: " << result.mean << " CPE +-" << result.stddev << std::endl;
}

template <typename RNGFunc>
void genThroughputTest(std::ostream& output = std::cerr)
{
    RNGFunc rng1(1), rng2(2), rng3(3), rng4(4);
    uint64_t acc1, acc2, acc3, acc4;
    std::vector<double> dataVec;

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

        uint32_t aux;
        uint64_t end = __rdtscp(&aux);
        _mm_lfence();

        dataVec.push_back((end - begin) / GEN_AMOUNT);
    }

    ResData result = detail::calculateStats(dataVec);
    output << "Throughput: " << result.mean << " CPE +-" << result.stddev << std::endl;
}

template <typename RNG>
void vGenThroughputTest(RNG& rng, const size_t& size = 1000000, std::ostream& output = std::cerr) {
    std::vector<float> res(size);
    std::vector<double> dataVec;

    for (int n = 0; n != ITERATIONS; ++n) {
        _mm_lfence();
        uint64_t begin = __rdtsc();
        _mm_lfence();

        rng.generateFloat(res); // questionable, might be rename func in minrandlib?

        uint32_t aux;
        uint64_t end = __rdtscp(&aux);
        _mm_lfence();

        volatile float dummy = res[size - 1];
        static_cast<void>(dummy);

        dataVec.push_back((end - begin) / size);
    }

    ResData result = detail::calculateStats(dataVec);
    output << "Vector RNG Throughput: " << result.mean << " CPE +- " << result.stddev << std::endl;
}

}  // namespace benchlib