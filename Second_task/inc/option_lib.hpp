#pragma once
#include "utils.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>
#include <random>

namespace financial {
struct OptionParameters {
    double spotPrice_;
    double strikePrice_;
    double timeToMaturity_;
    double riskFreeRate_;
    double volatility_;
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

inline std::ostream& operator<<(std::ostream& outStream, const OptionParameters& params) {
    outStream << "Option type:      " << params.optionType_     << "\n"
              << "Spot price:       " << params.spotPrice_      << "\n"
              << "Strike price:     " << params.strikePrice_    << "\n"
              << "Time to maturity: " << params.timeToMaturity_ << "\n"
              << "Risk free rating: " << params.riskFreeRate_   << "\n"
              << "Volatility:       " << params.volatility_;

    return outStream;
}

inline std::ostream& operator<<(std::ostream& outStream, const AmericanParameters params) {
    outStream << "Time step:       " << params.timeStep   << "\n"
              << "Up factor:       " << params.upFactor   << "\n"
              << "Down factor:     " << params.downFactor << "\n"
              << "Risk factor:     " << params.riskFactor << "\n"
              << "Discount factor: " << params.discountFactor;

    return outStream;
}
} // namespace financial

namespace finutils {
std::vector<financial::OptionParameters> dataGenerator();
} // namespace finutils