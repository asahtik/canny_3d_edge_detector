cmake_minimum_required(VERSION 3.0.0)
project(CannyEdgeDetector VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV 4.0.0 REQUIRED)

add_executable(CannyEdgeDetector main.cpp detector.hpp)

target_include_directories(CannyEdgeDetector PUBLIC include)
target_include_directories(CannyEdgeDetector PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(CannyEdgeDetector PUBLIC ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
