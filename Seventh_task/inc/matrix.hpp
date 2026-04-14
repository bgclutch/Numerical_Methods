#pragma once
#include <immintrin.h>

#include <cstdlib>
#include <random>
#include <vector>

#include "utils.hpp"

/*  448 KiB common L1d cache size
    For my Intel Core i7-1260P:
    8 e-cores with 32 KiB L1d cache
    4 p-cores with 48 KiB L1d cache
    I will implement optimised version for p-core case

    Also AVX512 is NOT available on my chip model, so
    I wrote preprocessing unit that chooses AVX512 if it is available
*/

#if defined(__AVX512F__)
#define VEC_WIDTH 16
#define ALIGNMENT 64
#define VEC_TYPE __m512
#define VEC_LOAD _mm512_load_ps
#define VEC_STORE _mm512_store_ps
#define VEC_SET1 _mm512_set1_ps
#define VEC_FMA _mm512_fmadd_ps
#else
#define VEC_WIDTH 8
#define ALIGNMENT 32
#define VEC_TYPE __m256
#define VEC_LOAD _mm256_loadu_ps
#define VEC_STORE _mm256_storeu_ps
#define VEC_SET1 _mm256_set1_ps
#define VEC_FMA _mm256_fmadd_ps
#endif

namespace matrix
{
static const int SEED    = 42;
static const int MIN_POW = 8;
static const int MAX_POW = 8;
static const int SRC_NUM = 3;

template <typename ElemType>
class MatrixSet
{
private:
    size_t firstRows;
    size_t firstCols;
    size_t secondCols;

    std::vector<ElemType> firstSrc;
    std::vector<ElemType> secondSrc;
    std::vector<ElemType> result;

private:
    std::vector<ElemType> fillMatrix(const size_t size)
    {
        std::mt19937 dataUni(SEED);
        std::uniform_real_distribution<ElemType> distUni(-100, 100);

        std::vector<ElemType> data(size);

        for (size_t i = 0; i != size; ++i) data[i] = distUni(dataUni);

        return data;
    }

    size_t setMatrixSize()
    {
        std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(MIN_POW, MAX_POW);
        return 1ull << dist(gen);
    }

public:
    MatrixSet()
        : firstRows{setMatrixSize()},
          firstCols{setMatrixSize()},
          secondCols{setMatrixSize()},
          firstSrc{fillMatrix(firstRows * firstCols)},
          secondSrc{fillMatrix(firstCols * secondCols)},
          result{std::vector<ElemType>(firstRows * secondCols, 0)}
    {
    }

    size_t getRows_1() const noexcept { return firstRows; }

    size_t getCols_1() const noexcept { return firstCols; }

    size_t getCols_2() const noexcept { return secondCols; }

    std::vector<ElemType> getResult() const { return result; }

    // 1. naive multiplication
    void naiveMult()
    {
        std::fill(result.begin(), result.end(), 0.0f);
        for (size_t i = 0; i != firstRows; ++i)
        {
            for (size_t j = 0; j != secondCols; ++j)
            {
                ElemType sum = 0;
                for (size_t k = 0; k != firstCols; ++k)
                {
                    sum += firstSrc[i * firstCols + k] * secondSrc[k * secondCols + j];
                }
                result[i * secondCols + j] = sum;
            }
        }
    }

    // 2. vectorised multiplication
    void vectMult()
    {
        std::fill(result.begin(), result.end(), 0.0f);
        for (size_t i = 0; i != firstRows; ++i)
        {
            for (size_t k = 0; k != firstCols; ++k)
            {
                ElemType factor = firstSrc[i * firstCols + k];
                for (size_t j = 0; j != secondCols; ++j)
                {
                    result[i * secondCols + j] += factor * secondSrc[k * secondCols + j];
                }
            }
        }
    }

    // 3. AVX multiplication
    void intrinsicMult()
    {
        std::fill(result.begin(), result.end(), 0.0f);
        for (size_t i = 0; i != firstRows; ++i)
        {
            for (size_t k = 0; k != firstCols; ++k)
            {
                VEC_TYPE vFirst = VEC_SET1(firstSrc[i * firstCols + k]);

                for (size_t j = 0; j != secondCols; j += VEC_WIDTH)
                {
                    VEC_TYPE vSecond = VEC_LOAD(&secondSrc[k * secondCols + j]);
                    VEC_TYPE vRes    = VEC_LOAD(&result[i * secondCols + j]);

                    vRes = VEC_FMA(vFirst, vSecond, vRes);
                    VEC_STORE(&result[i * secondCols + j], vRes);
                }
            }
        }
    }

    // 3.1 AVX multiplication improved version
    void intrinsicMultImproved()
    {
        std::fill(result.begin(), result.end(), 0.0f);
        for (size_t i = 0; i < firstRows; ++i)
        {
            for (size_t j = 0; j < secondCols; j += VEC_WIDTH)
            {
                VEC_TYPE vSum = VEC_SET1(0.0f);

                for (size_t k = 0; k < firstCols; ++k)
                {
                    VEC_TYPE vFirst  = VEC_SET1(firstSrc[i * firstCols + k]);
                    VEC_TYPE vSecond = VEC_LOAD(&secondSrc[k * secondCols + j]);
                    vSum             = VEC_FMA(vFirst, vSecond, vSum);
                }

                VEC_STORE(&result[i * secondCols + j], vSum);
            }
        }
    }

    // 3.2 AVX multiplication improved && tiled version
    void intrinsicMultTiled()
    {
        std::fill(result.begin(), result.end(), 0.0f);
        const size_t BS = 64;
        for (size_t ii = 0; ii < firstRows; ii += BS)
        {
            size_t i_end = std::min(ii + BS, firstRows);
            for (size_t kk = 0; kk < firstCols; kk += BS)
            {
                size_t k_end = std::min(kk + BS, firstCols);
                for (size_t jj = 0; jj < secondCols; jj += BS)
                {
                    size_t j_end = std::min(jj + BS, secondCols);
                    for (size_t i = ii; i < i_end; ++i)
                    {
                        for (size_t k = kk; k < k_end; ++k)
                        {
                            VEC_TYPE vA = VEC_SET1(firstSrc[i * firstCols + k]);
                            for (size_t j = jj; j < j_end; j += VEC_WIDTH)
                            {
                                VEC_TYPE vB = VEC_LOAD(&secondSrc[k * secondCols + j]);
                                VEC_TYPE vC = VEC_LOAD(&result[i * secondCols + j]);

                                vC = VEC_FMA(vA, vB, vC);
                                VEC_STORE(&result[i * secondCols + j], vC);
                            }
                        }
                    }
                }
            }
        }
    }

    // 3.3 AVX multiplication twice improved && tiled version
    void intrinsicMultAbsolute()
    {
        std::fill(result.begin(), result.end(), 0.0f);
        const size_t BS = 64;
        for (size_t ii = 0; ii < firstRows; ii += BS)
        {
            size_t i_end = std::min(ii + BS, firstRows);
            for (size_t kk = 0; kk < firstCols; kk += BS)
            {
                size_t k_end = std::min(kk + BS, firstCols);
                for (size_t jj = 0; jj < secondCols; jj += BS)
                {
                    size_t j_end = std::min(jj + BS, secondCols);
                    for (size_t i = ii; i < i_end; i += 2)
                    {
                        for (size_t j = jj; j < j_end; j += VEC_WIDTH * 4)
                        {
                            VEC_TYPE vC0  = VEC_LOAD(&result[i * secondCols + j + VEC_WIDTH * 0]);
                            VEC_TYPE vC1  = VEC_LOAD(&result[i * secondCols + j + VEC_WIDTH * 1]);
                            VEC_TYPE vC2  = VEC_LOAD(&result[i * secondCols + j + VEC_WIDTH * 2]);
                            VEC_TYPE vC3  = VEC_LOAD(&result[i * secondCols + j + VEC_WIDTH * 3]);
                            VEC_TYPE vC10 = VEC_LOAD(&result[(i + 1) * secondCols + j + VEC_WIDTH * 0]);
                            VEC_TYPE vC11 = VEC_LOAD(&result[(i + 1) * secondCols + j + VEC_WIDTH * 1]);
                            VEC_TYPE vC12 = VEC_LOAD(&result[(i + 1) * secondCols + j + VEC_WIDTH * 2]);
                            VEC_TYPE vC13 = VEC_LOAD(&result[(i + 1) * secondCols + j + VEC_WIDTH * 3]);

                            VEC_TYPE vB0 = VEC_LOAD(&secondSrc[kk * secondCols + j + VEC_WIDTH * 0]);
                            VEC_TYPE vB1 = VEC_LOAD(&secondSrc[kk * secondCols + j + VEC_WIDTH * 1]);
                            VEC_TYPE vB2 = VEC_LOAD(&secondSrc[kk * secondCols + j + VEC_WIDTH * 2]);
                            VEC_TYPE vB3 = VEC_LOAD(&secondSrc[kk * secondCols + j + VEC_WIDTH * 3]);

                            for (size_t k = kk; k < k_end; ++k)
                            {
                                VEC_TYPE vA0 = VEC_SET1(firstSrc[i * firstCols + k]);
                                vC0          = VEC_FMA(vA0, vB0, vC0);
                                vC1          = VEC_FMA(vA0, vB1, vC1);

                                VEC_TYPE vA1 = VEC_SET1(firstSrc[(i + 1) * firstCols + k]);

                                vC2 = VEC_FMA(vA0, vB2, vC2);
                                vC3 = VEC_FMA(vA0, vB3, vC3);

                                vC10 = VEC_FMA(vA1, vB0, vC10);
                                vC11 = VEC_FMA(vA1, vB1, vC11);

                                if (k + 1 < k_end)
                                {
                                    vB0 = VEC_LOAD(&secondSrc[(k + 1) * secondCols + j + VEC_WIDTH * 0]);
                                    vB1 = VEC_LOAD(&secondSrc[(k + 1) * secondCols + j + VEC_WIDTH * 1]);
                                }

                                vC12 = VEC_FMA(vA1, vB2, vC12);
                                vC13 = VEC_FMA(vA1, vB3, vC13);

                                if (k + 1 < k_end)
                                {
                                    vB2 = VEC_LOAD(&secondSrc[(k + 1) * secondCols + j + VEC_WIDTH * 2]);
                                    vB3 = VEC_LOAD(&secondSrc[(k + 1) * secondCols + j + VEC_WIDTH * 3]);
                                }
                            }

                            VEC_STORE(&result[i * secondCols + j + VEC_WIDTH * 0], vC0);
                            VEC_STORE(&result[i * secondCols + j + VEC_WIDTH * 1], vC1);
                            VEC_STORE(&result[i * secondCols + j + VEC_WIDTH * 2], vC2);
                            VEC_STORE(&result[i * secondCols + j + VEC_WIDTH * 3], vC3);
                            VEC_STORE(&result[(i + 1) * secondCols + j + VEC_WIDTH * 0], vC10);
                            VEC_STORE(&result[(i + 1) * secondCols + j + VEC_WIDTH * 1], vC11);
                            VEC_STORE(&result[(i + 1) * secondCols + j + VEC_WIDTH * 2], vC12);
                            VEC_STORE(&result[(i + 1) * secondCols + j + VEC_WIDTH * 3], vC13);
                        }
                    }
                }
            }
        }
    }

    // 4. OpenCL GPU multiplication
    void GPUMult()
    {
        ocl_utils::Environment env("matmul.cl", "matMul");

        cl::Context& context    = env.get_context();
        cl::CommandQueue& queue = env.get_queue();
        cl::Kernel& kernel      = env.get_kernel();

        size_t sizeA = firstRows * firstCols * sizeof(ElemType);
        size_t sizeB = firstCols * secondCols * sizeof(ElemType);
        size_t sizeC = firstRows * secondCols * sizeof(ElemType);

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeA, firstSrc.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeB, secondSrc.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeC);

        kernel.setArg(0, static_cast<int>(firstRows));
        kernel.setArg(1, static_cast<int>(firstCols));
        kernel.setArg(2, static_cast<int>(secondCols));
        kernel.setArg(3, bufferA);
        kernel.setArg(4, bufferB);
        kernel.setArg(5, bufferC);

        cl::NDRange global_work_size(secondCols, firstRows);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, cl::NullRange);
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeC, result.data());
    }
};
}  // namespace matrix