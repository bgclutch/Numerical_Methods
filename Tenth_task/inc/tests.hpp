#pragma once

#include <fstream>

namespace tests
{
void isRNGCorrect(std::ofstream& output);
void piBenchmark(std::ofstream& output);
}  // namespace tests