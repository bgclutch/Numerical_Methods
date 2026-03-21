#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <stdfloat>

namespace precision {
struct TestCase {
    size_t count;
    double mean;
    double stdDev;
};

using float_128 = long double;

template <typename ElemType>
class Solver {
 public:
    static ElemType fast(const std::vector<ElemType>& data) {
        ElemType RS = ElemType{0};
        ElemType M  = ElemType{0};
        auto size = static_cast<ElemType>(data.size());

        for (auto& elem : data) {
            M  += elem;
            RS += elem * elem;
        }

        RS /= size;
        M  /= size;

        ElemType DX = RS - M * M;

        if (DX < ElemType(0)) {
            std::cerr << "  [WARNING] Negative variance detected in Naive method! Value: " << DX << "\n";
        }

        return DX;
    }

    static ElemType twoPass(const std::vector<ElemType>& data) {
        ElemType M = 0;
        auto size = static_cast<ElemType>(data.size());

        for (auto& elem : data) {
            M += elem;
        }

        M /= size;
        ElemType DX = 0;

        for (auto& elem : data) {
            auto tmp = elem - M;
            DX += tmp * tmp;
        }

        DX /= size;
        return DX;
    }

    static ElemType singlePass(const std::vector<ElemType>& data) {
        ElemType DX = 0;
        ElemType M = 0;
        auto size = static_cast<ElemType>(data.size());

        for (size_t i = 1; i <= size; ++i) {
            ElemType x = data[i - 1];
            ElemType oldM = M;
            M += (x - M) / i;
            DX += ((x - oldM) * (x - M) - DX) / i;
        }
        return DX;
    }

    static float_128 calcReference(const std::vector<double>& data) {
        float_128 m = 0;
        float_128 s = 0;
        auto size = data.size();

        for (size_t i = 1; i <= size; ++i) {
            float_128 x = static_cast<float_128>(data[i - 1]);
            float_128 oldM = m;
            m += (x - m) / static_cast<float_128>(i);
            s += (x - oldM) * (x - m);
        }
        return s / static_cast<float_128>(data.size());
    }
};

template <typename ElemType>
void runTestCases(const std::vector<TestCase>& cases, std::ofstream& csv, const std::string& typeName) {
    std::mt19937 gen(42);

    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        std::normal_distribution<double> dist(c.mean, c.stdDev);

        std::vector<double> dataDbl(c.count);
        for (auto& x : dataDbl) {
            x = dist(gen);
        }

        std::vector<ElemType> dataTarget;
        dataTarget.reserve(dataDbl.size());
        for (double val : dataDbl) {
            dataTarget.push_back(static_cast<ElemType>(val));
        }

        double trueSampleVar = static_cast<float_128>(Solver<double>::calcReference(dataDbl));

        ElemType resFast = Solver<ElemType>::fast(dataTarget);
        ElemType resTwo  = Solver<ElemType>::twoPass(dataTarget);
        ElemType resOne  = Solver<ElemType>::singlePass(dataTarget);

        double errFast = std::abs((static_cast<double>(resFast) - trueSampleVar) / trueSampleVar);
        double errTwo  = std::abs((static_cast<double>(resTwo)  - trueSampleVar) / trueSampleVar);
        double errOne  = std::abs((static_cast<double>(resOne)  - trueSampleVar) / trueSampleVar);

        csv << i + 1 << ","
            << c.count << ","
            << typeName << ","
            << c.mean << ","
            << c.stdDev << ","
            << trueSampleVar << ","
            << errFast << ","
            << errTwo << ","
            << errOne << "\n";
    }
    csv << "\n";
}
} // namespace precision