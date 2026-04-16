#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <numbers>
#include <bit>
#include <cstdint>
#include <type_traits>

namespace rng {
/*  Q*
    This generator is a simple counter
    It will perfectly fill [0, 1] and pass KS-test perfectly too
    Howewer, it will lose autocorrelation test due to its non-random realisation
*/

class BadGenerator {
    double current = 0.0;
    double step;
public:
    BadGenerator(double s = 0.001) : step(s) {}
    template <typename Type>
    Type next() {
        if constexpr (std::is_floating_point_v<Type>) {
                current = std::fmod(current + step, 1.0);
                return current;
        }
        else if constexpr (std::is_integral_v<Type>) {
            current = std::fmod(current + step, 1.0);
            return current * std::numeric_limits<Type>::max();
        }
        else {
            throw std::runtime_error("unsupported type");
        }
    }
};

// Q1. Chi-squared
template <typename ElemType>
double chiSquaredTest(const std::vector<ElemType>& sample, size_t bins = 100) {
    std::vector<int> observed(bins, 0);
    size_t size = sample.size();
    for (ElemType x : sample) {
        size_t bin = static_cast<size_t>(x * bins);

        if (bin >= bins)
            bin = bins - 1;

        ++observed[bin];
    }

    double expected = static_cast<double>(size) / bins;
    double chi2 = 0;

    for (int count : observed) {
        double diff = count - expected;
        chi2 += (diff * diff) / expected;
    }

    return chi2;
}

// Q2. Autocorrelation
template <typename ElemType>
double autocorrelationTest(const std::vector<ElemType>& sample, size_t lag = 1) {
    size_t size = sample.size();
    double mean = std::accumulate(sample.begin(), sample.end(), 0.0) / size;
    double num = 0;
    double den = 0;

    for (size_t i = 0; i != size; ++i) {
        double diff = sample[i] - mean;
        den += diff * diff;

        if (i + lag < size) {
            num += diff * (sample[i + lag] - mean);
        }
    }

    return num / den;
}

#if 1
// Q2. Upgraded Autocorrelation
template <typename ElemType>
double autocorrelationTestUpd(const std::vector<ElemType>& sample, size_t lag = 1) {
    size_t elemBitSize   = sizeof(ElemType) * 8;
    size_t sampleBitSize = sample.size() * elemBitSize;

    size_t compsNum = sampleBitSize - lag;
    int sum = 0;

    auto getBit = [&](size_t i){ // getting needed bit from i_th element in sample
        size_t index = i / elemBitSize;
        size_t indexOffset = i % elemBitSize;

        return (sample[index] >> indexOffset) & 1;
    };

    for (size_t i = 0; i != compsNum; ++i) {
        #if 0
        size_t index_i = i / elemBitSize;
        size_t index_i_lag = (i + lag) / elemBitSize;

        size_t index_i_offset = i % elemBitSize;
        size_t index_i_lag_offset = (i + lag) % elemBitSize;

        int bit_i = (sample[index_i] >> index_i_offset) & 1;
        int bit_i_lag = (sample[index_i_lag] >> index_i_lag_offset) & 1;
        #endif
        #if 1
        int bit_i = getBit(i);
        int bit_i_lag = getBit(i + lag);
        #endif
        sum += bit_i ^ bit_i_lag;
    }

    /*  Результат нужно валидировать: в идеальном случае у нас коэффициент автокорреляции будет
        равен compsNum / 2 -- это наше матожидание.
        Дисперсия будет равна compsNum / 4, тк распределение биномиальное
        Стандартное отклонение это корень из дисперсии
        Тогда получим доверительный интервал, он должен быть не более трех стандартных отклонений (сигм)
        Далее этот интервал мы должны обработать и понять, действительно ли генератору можно доверять, то есть с
        какой вероятностью он работает хорошо (тут чем меньше число - тем ниже доверие)
    */

    double trustInt = std::abs(2. * static_cast<double>(sum) - static_cast<double>(compsNum)) / std::sqrt(compsNum);
    double res = std::erfc(trustInt / std::numbers::sqrt2);
    std::cerr << "sum: " << sum << "\n" << "trust interval: " << trustInt << "\n" << "result: " << res << "\n";
    return res;
}
#endif


// Q2. KS-test
template <typename ElemType>
double ksTest(std::vector<ElemType> sample) {
    std::sort(sample.begin(), sample.end());
    double max_diff = 0;
    double n = static_cast<double>(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) {
        double empirical_cdf = (i + 1) / n;
        double theoretical_cdf = static_cast<double>(sample[i]);
        max_diff = std::max(max_diff, std::abs(empirical_cdf - theoretical_cdf));
    }
    return max_diff;
}
} // namespace rng