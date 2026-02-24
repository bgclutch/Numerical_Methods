#pragma once
#include "option_lib.hpp"
#include "utils.hpp"
#include <algorithm>
#include <vector>
#include <cmath>

namespace financial {
class BinominalCalculation {
 private:
    std::vector<double> optionPrices_;
    size_t calcSteps_;
 public:
    explicit BinominalCalculation(size_t calcSteps) : calcSteps_(calcSteps) {
        optionPrices_.resize(calcSteps_ + 1);
    }

    double calcPrice(const OptionParameters& optionParams) {
        AmericanParameters americanParams(optionParams, calcSteps_);
        double priceChanger = americanParams.downFactor / americanParams.upFactor;
        double maxPrice = optionParams.spotPrice_ * std::pow(americanParams.upFactor, calcSteps_);

        auto tmp = maxPrice;
        for (size_t i = 0; i <= calcSteps_; ++i) {
            optionPrices_[i] = calcFinalPrice(optionParams, tmp);
            tmp *= priceChanger;
        }

        for (size_t curStep = calcSteps_; curStep --> 0;) {
            maxPrice /= americanParams.upFactor;
            tmp = maxPrice;

            for (size_t i = 0; i <= curStep; ++i) {
                auto holdValue = calcHoldPrice(americanParams, optionPrices_[i], optionPrices_[i + 1]);

                auto execPrice = tmp;
                auto execValue = calcFinalPrice(optionParams, execPrice);

                optionPrices_[i] = std::max(execValue, holdValue);
                tmp *= priceChanger;
            }
        }
        return optionPrices_[0];
    }

 private:
    double calcFinalPrice(const OptionParameters& optionParams, const double& payoff) {
        auto tmp = payoff - optionParams.strikePrice_;
        tmp = (optionParams.optionType_ == finutils::OptionType::Call) ? tmp : -tmp;
        return std::max(0., tmp);
    }

    double calcHoldPrice(const AmericanParameters& americanParams, const double nextUp, const double nextDown) {
        return americanParams.discountFactor * (americanParams.riskFactor * nextUp + (1 - americanParams.riskFactor) * nextDown);
    }
};

} // namespace financial