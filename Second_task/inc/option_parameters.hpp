#pragma once
#include "utils.hpp"
#include <cmath>
#include <algorithm>

namespace financial {
struct OptionParameters {
    double spotPrice_;
    double strikePrice_;
    double timeToMaturity_;
    double riskFreeRate_;
    double volatility_;
    bool   isCall;
    finutils::OptionType optionType_;
};

struct AmericanParameters {
    double timeStep       = 0.;
    double upFactor       = 0.;
    double downFactor     = 0.;
    double riskFactor     = 0.;
    double discountFactor = 0.;

    AmericanParameters(const OptionParameters& params, const int calcSteps) :
        timeStep(params.timeToMaturity_ / calcSteps),
        upFactor(std::exp(params.volatility_ * std::sqrt(timeStep))),
        downFactor(1 / upFactor),
        riskFactor((std::exp(params.riskFreeRate_ * timeStep) - downFactor) / (upFactor - downFactor)),
        discountFactor(std::exp(-params.riskFreeRate_ * timeStep)) {}
};

} // namespace utils