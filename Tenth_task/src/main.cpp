#include "minstdrand.hpp"
#include "tests.hpp"
#include <iostream>
#include <fstream>

int main() {
    std::ofstream file("minstdres.txt");
    file << "Correctness of std::minstd_rand\n";
    tests::isRNGCorrect(file);

    file << std::endl;

    file << "Parallel Pi Benchmark\n";
    tests::piBenchmark(file);

    return 0;
}