#include "rng_validation.hpp"
#include <iostream>
#include <fstream>
#include <random>

int main() {
    const size_t RUNS = 100;
    const size_t SAMPLE_SIZE = 10000;

    std::ofstream csv("rng_validation.csv");
    csv << "Generator,Run,Chi2,KS,AutoCorr\n";

    std::mt19937 mtGen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    rng::BadGenerator badGen;

    for (int run = 0; run != RUNS; ++run) {
        std::vector<double> mtSample(SAMPLE_SIZE);
        std::vector<double> badSample(SAMPLE_SIZE);
        std::vector<uint32_t> bitSample(SAMPLE_SIZE);
        std::vector<uint32_t> badBitSample(SAMPLE_SIZE);


        for (size_t i = 0; i != SAMPLE_SIZE; ++i) {
            mtSample[i]  = dist(mtGen);
            badSample[i] = badGen.next<double>();
            bitSample[i] = mtGen();
            badBitSample[i] = badGen.next<uint32_t>();
        }

        csv << "MT19937," << run << ","
            << rng::chiSquaredTest(mtSample) << ","
            << rng::ksTest(mtSample) << ","
            << rng::autocorrelationTestUpd(bitSample) << "\n";

        csv << "LowDiscrepancy," << run << ","
            << rng::chiSquaredTest(badSample) << ","
            << rng::ksTest(badSample) << ","
            << rng::autocorrelationTestUpd(badBitSample) << "\n";

    }

    return 0;
}