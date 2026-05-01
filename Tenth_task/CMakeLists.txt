cmake_minimum_required(VERSION 3.13)

project(minstd)

find_package(OpenMP REQUIRED)

add_executable(minstd)

option(ENABLE_ASAN OFF)

target_compile_features(minstd PUBLIC cxx_std_23)

target_link_libraries(minstd PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(minstd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_sources(minstd PRIVATE
    src/main.cpp
    src/tests.cpp
)

target_compile_options(minstd PRIVATE
    -fopenmp
$<$<CONFIG:Debug>:
    -Wall
    -Wextra
    -Wpedantic
    -DDEBUG
    -Wmissing-declarations
    -Wcast-align
    -Wshadow
    -Wsign-conversion>
)

if(ENABLE_ASAN)
    target_compile_options(minstd PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
    target_link_options(minstd PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
endif()