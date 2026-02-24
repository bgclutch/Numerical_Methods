#include "american_option.hpp"
#include "option_lib.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

int main() {
    std::vector<financial::OptionParameters> data = finutils::dataGenerator();
    std::vector<double> prices(finutils::OPTIONS_AMOUNT);

    auto begin = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        financial::BinominalCalculation calcOpt(finutils::DEFAULT_STEPS);
        #pragma omp for schedule(static)
        for (auto i = 0; i < finutils::OPTIONS_AMOUNT; ++i) {
            prices[i] = calcOpt.calcPrice(data[i]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "execution time: " << time << " ms\n";

    std::ofstream writeFile;
    writeFile.open("options.txt");

    for (size_t i = 0; i < finutils::OPTIONS_AMOUNT; ++i) {
        writeFile << "Best price: " << prices[i] << " for\n" << data[i] << "\n\n";
    }

    writeFile.close();

    return EXIT_SUCCESS;
}