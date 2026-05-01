cmake_minimum_required(VERSION 3.13)

project(option)

find_package(OpenMP REQUIRED)

add_executable(option)
add_executable(tests)

option(ENABLE_ASAN OFF)

foreach(target option tests)
    target_compile_features(${target} PUBLIC cxx_std_23)
endforeach()

target_link_libraries(option PRIVATE
    OpenMP::OpenMP_CXX
)

target_link_libraries(tests PRIVATE
    OpenMP::OpenMP_CXX
    sleef
)

foreach(target option tests)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )
endforeach()

target_include_directories(tests PRIVATE
        ${CMAKE_SOURCE_DIR}/../Log_task/inc
        ${CMAKE_SOURCE_DIR}/../Tenth_task/inc
    )

target_sources(option PRIVATE
    src/main.cpp
    src/data_generator.cpp
    src/european_option.cpp
)

target_sources(tests PRIVATE
    src/european_option.cpp
    src/data_generator.cpp
    src/bsoptimisation.cpp
    src/bsmontecarlo.cpp
    src/tests.cpp
    ${CMAKE_SOURCE_DIR}/../Log_task/src/loglib.cpp
)

foreach(target option tests)
    target_compile_options(${target} PRIVATE
        -fopenmp
        -mfma
        -mavx2
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
endforeach()

foreach(target option tests)
    if(ENABLE_ASAN)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
        target_link_options(${target} PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
    endif()
endforeach()