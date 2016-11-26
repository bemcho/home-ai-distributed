cmake_minimum_required(VERSION 3.0)
project(home_ai_distributed)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -m64")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CAF_ROOT_DIR "/usr/local/")

find_package(Boost REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

find_package(CAF COMPONENTS  REQUIRED)
if(CAF_FOUND)
    include_directories(${CAF_INCLUDE_DIRS})
endif()

set(SOURCE_FILES source/main.cpp)
add_executable(home_ai_distributed ${SOURCE_FILES})
set(LIBS ${LIBS} ${OpenCV_LIBS} ${CAF_LIBRARIES} ${Boost_LIBS})
target_link_libraries( home_ai_distributed  ${LIBS})

