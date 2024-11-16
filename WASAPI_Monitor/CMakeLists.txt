cmake_minimum_required(VERSION 3.15)
project(onnx_inference_opencv)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the OpenCV directory path
set(OpenCV_DIR "C:/Users/rober/opencv/build/x64/vc15")  # Replace vc15 with vc16 if using Visual Studio 2019
find_package(OpenCV 4.10.0 REQUIRED)


# Add executable
add_executable(inference main.cpp)

# Link to the OpenCV world library
target_link_libraries(inference PRIVATE ${OpenCV_LIBS} opencv_world4100)


