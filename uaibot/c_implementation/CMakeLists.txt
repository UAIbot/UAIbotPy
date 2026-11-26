cmake_minimum_required(VERSION 3.18)
project(UAIBOT_CPP_BINDS LANGUAGES C CXX)

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
# Honour an explicitly selected pybind11 first; otherwise query this Python.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        RESULT_VARIABLE PYBIND11_QUERY_RESULT
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        ERROR_VARIABLE PYBIND11_QUERY_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT PYBIND11_QUERY_RESULT EQUAL 0)
        message(FATAL_ERROR "Install pybind11 for ${Python3_EXECUTABLE}: ${PYBIND11_QUERY_ERROR}")
    endif()
    find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)
endif()

# Prefer a system/configured Eigen, retaining the repository's bundled fallback.
find_package(Eigen3 3.3 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
    set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party" CACHE PATH
        "Directory containing Eigen/Core")
    if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
        message(FATAL_ERROR "Eigen not found: set Eigen3_DIR or EIGEN3_INCLUDE_DIR")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

# Avoid automatic LTO and its memory cost for Eigen translation units.
# Keep the top-level module name used by existing Python callers.
pybind11_add_module(uaibot_cpp_bind NO_EXTRAS
    pybind_main.cpp manipulator.cpp utils.cpp gjk.c smooth_functions.cpp
    dynamics_bindings.cpp auv_bindings.cpp uav_bindings.cpp
    manipulator_bindings.cpp robot_bindings.cpp
)
# Bundled Eigen 3.3.7 relies on pre-C++20 comparison overload resolution.
# cxx_std_17 would allow a compiler that defaults to C++20 or newer.
set_target_properties(uaibot_cpp_bind PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_include_directories(uaibot_cpp_bind PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(uaibot_cpp_bind PRIVATE Eigen3::Eigen pybind11::windows_extras)
target_compile_definitions(uaibot_cpp_bind PRIVATE EIGEN_MPL2_ONLY)
if(MSVC)
    target_compile_definitions(uaibot_cpp_bind PRIVATE _USE_MATH_DEFINES _HAS_STD_BYTE=0)
    target_compile_options(uaibot_cpp_bind PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        /wd4244 /wd4267 /wd4996)
else()
    target_compile_options(uaibot_cpp_bind PRIVATE -Wno-deprecated-declarations)
endif()

if(APPLE)
    set_target_properties(uaibot_cpp_bind PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(uaibot_cpp_bind PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
set_target_properties(uaibot_cpp_bind PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Tests import the extension directly, without Uaibot Python wrappers.
option(UAIBOT_BUILD_DYNAMICS_TESTS "Register the dynamics binding regression suite" OFF)
if(UAIBOT_BUILD_DYNAMICS_TESTS)
    enable_testing()
    add_test(NAME dynamics_bindings
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dynamics_bindings.py"
                "$<TARGET_FILE_DIR:uaibot_cpp_bind>")
endif()

# Optional comparison with an external snapshot; never required to build Uaibot.
set(UAIBOT_LEGACY_HEADERS_DIR "" CACHE PATH "Original dynamics header snapshot")
if(UAIBOT_BUILD_DYNAMICS_TESTS AND UAIBOT_LEGACY_HEADERS_DIR)
    foreach(variant IN ITEMS current legacy)
        add_executable(dynamics_${variant}_trajectory tests/test_dynamics_legacy.cpp)
        set_target_properties(dynamics_${variant}_trajectory PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        target_link_libraries(dynamics_${variant}_trajectory PRIVATE Eigen3::Eigen)
    endforeach()
    target_include_directories(dynamics_current_trajectory PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(dynamics_legacy_trajectory PRIVATE "${UAIBOT_LEGACY_HEADERS_DIR}")
    add_test(NAME dynamics_legacy_regression
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dynamics_legacy.py"
                "$<TARGET_FILE:dynamics_current_trajectory>" "$<TARGET_FILE:dynamics_legacy_trajectory>")
endif()
