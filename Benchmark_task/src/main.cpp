#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "benchlib.hpp"

int main()
{
    constexpr size_t size = 10000;
    std::vector<float> data(size);
    std::mt19937 gen(42);

    std::generate(data.begin(), data.end(), gen);
    auto stdlog = [](double x) { return std::log(x); };

    auto stdexp = [](double x) { return std::exp(x); };

    std::cout << "stdlog res: "
              << "\n";
    benchlib::funcLatencyTest(stdlog, data);
    benchlib::funcThroughputTest(stdlog, data);
    std::cout << "-------------------------------\n";

    std::cout << "stdexp res: "
              << "\n";
    benchlib::funcLatencyTest(stdexp, data);
    benchlib::funcThroughputTest(stdexp, data);
    std::cout << "-------------------------------\n";

    return EXIT_SUCCESS;
}