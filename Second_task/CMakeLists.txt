cmake_minimum_required(VERSION 3.13)

project(option)
add_executable(option)

option(ENABLE_ASAN OFF)

target_compile_features(option PUBLIC cxx_std_23)

target_include_directories(option PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_sources(option PRIVATE
    src/main.cpp
    src/data_generator.cpp
)

target_compile_options(option PRIVATE
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
    target_compile_options(option PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
    target_link_options(option PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
endif()