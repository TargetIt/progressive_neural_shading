cmake_minimum_required(VERSION 3.20)
project(phase11_1_coopvec)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This is a simplified CMakeLists.txt for Phase 11.1 (CoopVec)
# The full project requires external dependencies:
#   - Slang (shader compiler)
#   - slang-rhi (render hardware interface)
#   - NVIDIA cooperative vector support
#
# See the reference hardware-acceleration/ directory for the full build.

file(GLOB_RECURSE PROJECT_SOURCES
    mlp-training-coopvec/*.cpp
    mlp-training-coopvec/*.h
)

message(STATUS "Phase 11.1 source files:")
foreach(F ${PROJECT_SOURCES})
    message(STATUS "  ${F}")
endforeach()

add_custom_target(check_phase11_1_structure
    COMMAND ${CMAKE_COMMAND} -E echo "Phase 11.1 project structure verified"
    COMMENT "Verifying project structure"
)
