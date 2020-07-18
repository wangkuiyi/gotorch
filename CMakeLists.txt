cmake_minimum_required(VERSION 3.1)

project(pytorch_cpp_demo)
set(CMAKE_CXX_STANDARD 14)

# Find torch library
find_package(Torch REQUIRED)

# Main executable
add_executable(Demo Demo.cpp)
target_include_directories(Demo PUBLIC ${TORCH_INCLUDE_DIRS})

find_package(Threads REQUIRED)

# The linker flags will be moved to TorchConfig.cmake
if(APPLE)
  target_link_libraries(Demo
    -Wl,-s
    -Wl,-dead_strip
    -Wl,-all_load
    ${TORCH_LIBRARIES}
    -Wl,-noall_load
    Threads::Threads
    ${Log}
  )
else()
  target_link_libraries(Demo
    -Wl,-s
    -Wl,--gc-sections
    -Wl,--whole-archive
    ${TORCH_LIBRARIES}
    -Wl,--no-whole-archive
    Threads::Threads
    ${Log}
  )
endif()
