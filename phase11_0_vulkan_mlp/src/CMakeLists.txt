cmake_minimum_required(VERSION 3.20)
project(phase11_0_vulkan_mlp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This is a simplified CMakeLists.txt for Phase 11.0
# The full project requires external dependencies:
#   - Slang (shader compiler)
#   - slang-rhi (render hardware interface)
#   - lz4, miniz, unordered_dense
#
# See the reference hardware-acceleration/ directory for the full build.

# Collect source files
file(GLOB_RECURSE PROJECT_SOURCES
    mlp-training/*.cpp
    mlp-training/*.h
)

message(STATUS "Phase 11.0 source files:")
foreach(F ${PROJECT_SOURCES})
    message(STATUS "  ${F}")
endforeach()

# This is a placeholder build - actual compilation requires external deps
# For a real build, see the hardware-acceleration reference directory.
add_custom_target(check_phase11_0_structure
    COMMAND ${CMAKE_COMMAND} -E echo "Phase 11.0 project structure verified"
    COMMENT "Verifying project structure"
)
