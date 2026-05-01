#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "american_option.hpp"
#include "option_lib.hpp"
#include "utils.hpp"

int main()
{
    std::vector<financial::OptionParameters> data = finutils::dataGenerator();
    std::vector<double> americanPrices(finutils::OPTIONS_AMOUNT);
    std::vector<double> europeanPrices(finutils::OPTIONS_AMOUNT);

#pragma omp parallel
    {
        financial::BinominalCalculation calcOpt(finutils::DEFAULT_STEPS);
#pragma omp for schedule(static)
        for (size_t i = 0; i < finutils::OPTIONS_AMOUNT; ++i)
        {
            americanPrices[i] = calcOpt.calcPrice(data[i]);
            europeanPrices[i] = calcBSPrice(data[i]);
        }
    }

    std::ofstream writeFile;
    writeFile.open("options.txt");

    for (size_t i = 0; i < finutils::OPTIONS_AMOUNT; ++i)
    {
        writeFile << "Best price for American (via Binomial): " << americanPrices[i]
                  << "\nPrice for European (via Black-Scholes): " << europeanPrices[i] << "\n"
                  << data[i] << "\n\n";
    }

    writeFile.close();

    return EXIT_SUCCESS;
}