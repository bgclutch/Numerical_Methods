cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 23)

project(validation)
add_executable(validation)

option(ENABLE_ASAN OFF)

target_compile_features(validation PUBLIC cxx_std_23)

target_include_directories(validation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_sources(validation PRIVATE
    src/main.cpp
)

target_compile_options(validation PRIVATE
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
    target_compile_options(validation PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
    target_link_options(validation PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
endif()