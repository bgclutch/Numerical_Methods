#include <fstream>
#include <iostream>

#include "benchlib.hpp"
#include "minstdrand.hpp"
#include "tests.hpp"

int main()
{
    std::ofstream file("minstdres.txt");
    file << "Correctness of std::minstd_rand\n";
    tests::isRNGCorrect(file);

    file << std::endl;

    file << "Parallel Pi Benchmark\n";
    tests::piBenchmark(file);

    rng::VectorMinstd rng;
    rng.seed(42);
    std::cerr << "Generator Througput\n";
    benchlib::vGenThroughputTest(rng, 10000000);

    return EXIT_SUCCESS;
}