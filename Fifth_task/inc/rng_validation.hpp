#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

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
    double next() {
        current = std::fmod(current + step, 1.0);
        return current;
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