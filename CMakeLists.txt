cmake_minimum_required(VERSION 3.22)
project(computer_vision)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV CONFIG REQUIRED)

add_executable(cv main.cpp)

target_link_libraries(cv PRIVATE ${OpenCV_LIBS})