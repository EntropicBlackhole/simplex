cmake_minimum_required(VERSION 3.20)
project(simplex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core geometry videoio imgproc dnn)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(simplex src/main.cpp)

target_link_libraries(simplex PRIVATE
    ${OpenCV_LIBS}
    glfw
    OpenGL::GL
)

target_compile_options(simplex PRIVATE -Wall -Wextra -O3 -march=tigerlake -ffast-math)