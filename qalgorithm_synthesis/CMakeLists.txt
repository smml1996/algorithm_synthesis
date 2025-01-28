

cmake_minimum_required(VERSION 3.18)
project(qalgorithm_synthesis)

set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

# Add the source directory where src files are
include_directories(${PROJECT_SOURCE_DIR})

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(qalgorithm_synthesis main.cpp)


