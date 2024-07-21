

cmake_minimum_required(VERSION 3.18)
project(qalgorithm_synthesis)

set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

#target_link_libraries(nlohmann_json::nlohmann_json)
add_executable(qalgorithm_synthesis main.cpp)
