cmake_minimum_required(VERSION 3.27)
project(ObjectDetectionPipeline CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(onnxruntime REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(detector
    src/main.cpp
    src/detector.cpp
    src/preprocess.cpp
    src/postprocess.cpp
)
target_link_libraries(detector PRIVATE onnxruntime::onnxruntime ${OpenCV_LIBS})

add_subdirectory(tests)
