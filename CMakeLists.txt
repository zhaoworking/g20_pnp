cmake_minimum_required(VERSION 3.10)
project(BA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")
message("Mode",${CMAKE_BUILD_TYPE})
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(glog REQUIRED)


include_directories("/usr/local/include")
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS}
                    ${Eigen3_INCLUDE_DIRS})

# link_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(BA pnpg2o.cpp)
target_link_libraries(BA ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} 
                    g2o_core g2o_stuff g2o_types_sba g2o_types_slam3d 
                    # g2o_csparse_extension g2o_solver_csparse
                    # g2o_solver_cholmod
                    fmt::fmt glog)
                        
                    