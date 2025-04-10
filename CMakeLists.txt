cmake_minimum_required(VERSION 3.5.0)
project(code VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )
message("OpenCV include dir ${OpenCV_INCLUDE_DIRS}")


file(GLOB src *.cpp *.hpp)
add_executable(code ${src})
target_include_directories(code PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(code PUBLIC ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
