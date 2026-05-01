#include <vector>

#include "option_lib.hpp"
#include "utils.hpp"

namespace finutils
{
std::vector<financial::OptionParameters> dataGenerator()
{
    std::vector<financial::OptionParameters> data;
    data.reserve(OPTIONS_AMOUNT);

    std::mt19937 seed(42);
    std::uniform_real_distribution<double> spotGen(100.0, 200.0);
    std::uniform_real_distribution<double> strikeGenFactor(0.8, 1.2);
    std::uniform_real_distribution<double> timeGen(1.0, 1.29);
    std::uniform_real_distribution<double> riskFreeGen(0.02, 0.05);
    std::uniform_real_distribution<double> volatilityGen(0.15, 0.3);

    for (auto i = 0; i < OPTIONS_AMOUNT; ++i)
    {
        auto spotPrice           = spotGen(seed);
        OptionType optionTypeGen = (i % 2 == 0) ? OptionType::Call : OptionType::Put;
        data.push_back(
            {spotPrice, spotPrice * strikeGenFactor(seed), timeGen(seed), riskFreeGen(seed), volatilityGen(seed), optionTypeGen});
    }
    return data;
}
}  // namespace finutils