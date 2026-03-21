#include "test_utils.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdfloat>

int main() {
    std::ofstream csv("precision_results.csv");
    csv << "Number,Size,Type,Mean,StdDev,TrueSampleVar,ErrFast,ErrTwoPass,ErrSinglePass\n";

    std::vector<precision::TestCase> data = {
        {1000, 1.0, 1.0},
        {1000, 10.0, 10.0},
        {1000, 10.0, 0.1},
        {1000, 10000.0, 100.0},
        {1000, 100.0, 0.01},
        {1000, 1000.0, 1.0},
        {1000, 10.0, 0.001},
        {1000, 1.0, 0.0001},
        {1000, 100000.0, 10.0},
        {1000, 1000.0, 0.001},
    };

    precision::runTestCases<std::float32_t>(data, csv, "float32");
    precision::runTestCases<std::float64_t>(data, csv, "float64");

    return EXIT_SUCCESS;
}