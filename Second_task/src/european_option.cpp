#include <cmath>

#include "option_lib.hpp"

namespace financial
{
double calcCDF(const double num)
{
    return std::erfc(-num / finutils::SQRT_2) * 0.5;
}

double calcBSPrice(const OptionParameters& option)
{
    double volSqrtTime = option.volatility_ * std::sqrt(option.timeToMaturity_);
    double y1          = (std::log(option.spotPrice_ / option.strikePrice_) +
                 option.timeToMaturity_ * (option.riskFreeRate_ + option.volatility_ * option.volatility_ * 0.5)) /
                volSqrtTime;

    double y2    = y1 - volSqrtTime;
    double price = 0.;

    if (option.optionType_ == finutils::OptionType::Call)
    {
        price =
            option.spotPrice_ * calcCDF(y1) - option.strikePrice_ * std::exp(-option.riskFreeRate_ * option.timeToMaturity_) * calcCDF(y2);
    }
    else
    {
        price = -option.spotPrice_ * (1 - calcCDF(y1)) +
                option.strikePrice_ * std::exp(-option.riskFreeRate_ * option.timeToMaturity_) * (1 - calcCDF(y2));
    }

    return price;
}
}  // namespace financial