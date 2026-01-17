cmake_minimum_required(VERSION 3.20)
project(d3x LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Core library sources
set(D3X_SOURCES
    core/src/world.cpp
    core/src/gravity.cpp
    core/src/integrators.cpp
)

# Python extension module
pybind11_add_module(_core
    ${D3X_SOURCES}
    core/bindings/module.cpp
)

target_include_directories(_core PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)

# Install the extension module to the Python package
install(TARGETS _core LIBRARY DESTINATION d3x)

# Stub generation (run manually or via post-build)
find_package(Python COMPONENTS Interpreter REQUIRED)

add_custom_target(stubs
    COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen d3x._core
            --output-dir ${CMAKE_SOURCE_DIR}/src
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS _core
    COMMENT "Generating Python type stubs"
)

# C++ tests (optional, requires doctest)
option(BUILD_TESTS "Build C++ tests" OFF)

if(BUILD_TESTS)
    enable_testing()

    # Fetch doctest
    include(FetchContent)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.4.11
    )
    FetchContent_MakeAvailable(doctest)

    # Test executable
    add_executable(d3x_tests
        ${D3X_SOURCES}
        tests/cpp/test_main.cpp
        tests/cpp/test_orbits.cpp
    )

    target_include_directories(d3x_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )

    target_link_libraries(d3x_tests PRIVATE doctest::doctest)

    add_test(NAME d3x_tests COMMAND d3x_tests)
endif()
