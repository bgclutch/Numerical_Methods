#include <omp.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "loglib.hpp"
#include "option_lib.hpp"
#include "utils.hpp"

namespace tests
{

void testTask13()
{
    std::cout << "\nTASK 13 Euler-Maruyama SIMD Verification\n";

    financial::OptionParameters opt = {100.0, 100.0, 0.05, 0.2, 1.0, finutils::OptionType::Call};

    double analytical = financial::calcBSPrice(opt);
    size_t paths      = 1000000;

    std::cout << "Reference Analytical Price: " << std::fixed << std::setprecision(6) << analytical << std::endl;
    std::cout << std::setw(10) << "Steps" << std::setw(15) << "MC Price" << std::setw(15) << "Abs Error" << std::setw(12) << "ms"
              << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    std::vector<uint64_t> testSteps = {10, 50, 100, 500};

    for (auto steps : testSteps)
    {
        double callMC = 0, putMC = 0;

        double start = omp_get_wtime();
        financial::calcSimdBS(opt, paths, steps, callMC, putMC);
        double end = omp_get_wtime();

        double error = std::abs(callMC - analytical);
        std::cout << std::setw(10) << steps << std::setw(15) << callMC << std::setw(15) << error << std::setw(12) << (end - start) * 1000.0
                  << std::endl;
    }
}

void testTask14()
{
    std::cout << "\nTASK 14 Monte-Carlo (100 Options)\n";

    constexpr size_t N_OPTS = 100;
    constexpr size_t PATHS  = 1000000;

    std::vector<financial::OptionParameters> portfolio = finutils::dataGenerator();

    std::vector<financial::MCResult> results(N_OPTS);

    std::vector<double> callRef(N_OPTS);
    for (size_t i = 0; i < N_OPTS; ++i)
    {
        portfolio[i].optionType_ = finutils::OptionType::Call;
        callRef[i]               = financial::calcBSPrice(portfolio[i]);
    }

    double start = omp_get_wtime();
    financial::calcOptionsMC(portfolio, PATHS, results);
    double end = omp_get_wtime();

    double max_err        = 0;
    double avg_err        = 0;
    uint64_t failed_count = 0;
    double tolerance      = 0.05;

    for (size_t i = 0; i < N_OPTS; ++i)
    {
        if (std::isnan(results[i].callPrice))
        {
            ++failed_count;
            continue;
        }

        double err = std::abs(results[i].callPrice - callRef[i]);
        max_err    = std::max(max_err, err);
        avg_err += err;
        if (err > tolerance) ++failed_count;
    }

    std::cout << "Portfolio Size   : " << N_OPTS << " options" << std::endl;
    std::cout << "Paths per Option : " << PATHS << std::endl;
    std::cout << "Execution Time   : " << (end - start) * 1000.0 << " ms" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Max Abs Error    : " << max_err << std::endl;
    std::cout << "Avg Abs Error    : " << (N_OPTS > 0 ? avg_err / N_OPTS : 0) << std::endl;
    std::cout << "Success Rate     : " << N_OPTS - failed_count << "/" << N_OPTS << std::endl;
    std::cout << "Result           : " << (failed_count == 0 ? "PASSED" : "FAILED") << std::endl;
}

}  // namespace tests

int main()
{
    math::detail::initLookUpTables();

    tests::testTask13();
    tests::testTask14();

    return EXIT_SUCCESS;
}