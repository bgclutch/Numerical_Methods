cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 23)

project(benchlib)
add_executable(benchlib)

option(ENABLE_ASAN OFF)

target_compile_features(benchlib PUBLIC cxx_std_23)

target_include_directories(benchlib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_sources(benchlib PRIVATE
    src/main.cpp
)

target_compile_options(benchlib PRIVATE
    -O3
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
    target_compile_options(benchlib PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
    target_link_options(benchlib PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
endif()