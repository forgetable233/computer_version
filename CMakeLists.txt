cmake_minimum_required(VERSION 3.22)
project(computer_vision)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_executable(cv main.cpp image_processor.cpp image_processor.h)

target_link_libraries(cv PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)