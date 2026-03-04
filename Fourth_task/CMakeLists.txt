cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 23)

project(monte_carlo)
add_executable(monte_carlo)

option(ENABLE_ASAN OFF)

target_compile_features(monte_carlo PUBLIC cxx_std_23)

target_include_directories(monte_carlo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_sources(monte_carlo PRIVATE
    src/main.cpp
)

target_compile_options(monte_carlo PRIVATE
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
    target_compile_options(monte_carlo PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
    target_link_options(monte_carlo PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
endif()