cmake_minimum_required(VERSION 3.10)
project(UAIBOT_CPP_BINDS)

# Windows-specific setup
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)  # Export all symbols for DLL
endif()
# Add python 3.11
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Pybind11 discovery with cross-platform support
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c 
        "import pybind11, os; print(os.path.dirname(pybind11.__file__))"
    OUTPUT_VARIABLE PYBIND11_ROOT_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "[DEBUG] Pybind11 root directory: ${PYBIND11_ROOT_DIR}")

# Set pybind11 paths
set(PYBIND11_INCLUDE_DIR "${PYBIND11_ROOT_DIR}/include")
set(pybind11_DIR "${PYBIND11_ROOT_DIR}/share/cmake/pybind11")
list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
include_directories(${PYBIND11_INCLUDE_DIR})
# Find pybind11 package
find_package(pybind11 CONFIG REQUIRED)

message(STATUS "[DEBUG] Python includes: ${Python3_INCLUDE_DIRS}")
message(STATUS "[DEBUG] Python library: ${Python3_LIBRARIES}")
message(STATUS "[DEBUG] Python executable: ${Python3_EXECUTABLE}")
message(STATUS "[DEBUG] Python lib directory: ${Python3_LIBRARY_DIR}")

#Add eigen
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(EIGEN3_INCLUDE_DIR ${THIRD_PARTY_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})


# Set C++ standard and optimization flags
set(CMAKE_CXX_STANDARD 14)
if(MSVC)
    # Enable math constants (M_PI etc.) and strict standards compliance
    message(STATUS "[DEBUG] Using MSVC")
    add_compile_definitions(_USE_MATH_DEFINES _HAS_STD_BYTE=0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /EHsc /wd4244 /wd4267 /O2 /DEIGEN_MPL2_ONLY /wd4996 /permissive- /D_USE_MATH_DEFINES")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DEIGEN_MPL2_ONLY -Wno-deprecated-declarations")
endif()

# Define source files 
set(SOURCES
    pybind_main.cpp   
    manipulator.cpp  
    utils.cpp  
    declarations.h  
    gjk.c         
    gjk.h
    smooth_functions.cpp
)

# Define an INTERFACE library for nanoflann (header-only)
add_library(nanoflann INTERFACE)
target_include_directories(nanoflann INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Ensure pybind11 and Eigen are properly included
pybind11_add_module(uaibot_cpp_bind ${SOURCES})

target_link_libraries(uaibot_cpp_bind PRIVATE
    pybind11::pybind11      # Python headers + core
    pybind11::windows_extras  # Windows-specific optimizations
)

target_include_directories(uaibot_cpp_bind PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(uaibot_cpp_bind PRIVATE nanoflann)
# Windows-specific linking
if(WIN32)
    target_link_libraries(uaibot_cpp_bind PRIVATE ${Python3_LIBRARIES})
endif()

# SET RUNTIME SEARCH PATH (Ensures the `.so` file finds dependencies)
set_target_properties(uaibot_cpp_bind PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(uaibot_cpp_bind PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(uaibot_cpp_bind PROPERTIES INSTALL_RPATH "$ORIGIN")

# Ensure gjk.c is compiled as C
set_source_files_properties(gjk.c PROPERTIES LANGUAGE C)
