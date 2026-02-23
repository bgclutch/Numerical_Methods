#include "american_option.hpp"
#include "option_parameters.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    std::vector<financial::OptionParameters> data = finutils::dataGenerator();
    std::vector<double> prices;
    prices.reserve(finutils::OPTIONS_AMOUNT);

    for (auto& elem : data) {
        financial::BinominalCalculation calcOpt(finutils::DEFAULT_STEPS);

        prices.push_back(calcOpt.calcPrice(elem));
    }

    std::ofstream writeFile;
    writeFile.open("options.txt");

    for (size_t i = 0; i < finutils::OPTIONS_AMOUNT; ++i) {
        writeFile << "Best price: " << prices[i] << " for\n" << data[i] << "\n" << std::endl;
    }

    writeFile.close();

    return EXIT_SUCCESS;
}