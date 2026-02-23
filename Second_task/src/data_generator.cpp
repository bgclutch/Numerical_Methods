#include "option_parameters.hpp"
#include "utils.hpp"
#include <vector>

namespace finutils {
std::vector<financial::OptionParameters> dataGenerator()  {
    std::vector<financial::OptionParameters> data;
    data.reserve(OPTIONS_AMOUNT);

    std::mt19937 seed(42);
    std::uniform_real_distribution<double> spotGen(10.0, 1000.0);
    std::uniform_real_distribution<double> strikeGenFactor(0.5, 2.7);
    std::uniform_real_distribution<double> timeGen(0.1, 10.0);
    std::uniform_real_distribution<double> riskFreeGen(0.01, 0.2);
    std::uniform_real_distribution<double> volatilityGen(0.05, 0.9);

    for (auto i = 0; i < OPTIONS_AMOUNT; ++i) {
        auto spotPrice = spotGen(seed);
        OptionType optionTypeGen = (i % 2 == 0) ? OptionType::Call : OptionType::Put;
        data.push_back({
            spotPrice,
            spotPrice * strikeGenFactor(seed),
            timeGen(seed),
            riskFreeGen(seed),
            volatilityGen(seed),
            optionTypeGen
        });
    }
    return data;
}
} // namespace finutils