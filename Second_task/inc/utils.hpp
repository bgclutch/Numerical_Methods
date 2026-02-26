#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

namespace finutils {
static const int OPTIONS_AMOUNT = 1000;
static const int DEFAULT_STEPS  = 1000;
static const double SQRT_2 = std::sqrt(2);

enum class OptionType {
    Call,
    Put
};

inline std::ostream& operator<<(std::ostream& outStream, const finutils::OptionType type) {
    std::string res = (type == finutils::OptionType::Call) ? "Call" : "Put";
    outStream << res;
    return outStream;
}
} // namespace finutils