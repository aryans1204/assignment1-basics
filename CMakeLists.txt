cmake_minimum_required(VERSION 3.15)
project(bpe_tokenizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(bpe_tokenizer cs336_basics/bpe_tokenizer.cpp)
