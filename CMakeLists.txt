cmake_minimum_required(VERSION 3.15)
project(plasma)

enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

option(PLASMA_WITH_MKL "Enable MKL" ON)
if (${PLASMA_WITH_MKL})
    find_package(MKL QUIET)
endif ()
if (NOT ${PLASMA_WITH_MKL} OR NOT ${MKL_FOUND})
    message(STATUS "MKL NOT ENABLED")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_library(LAPACKE_LIB lapacke)
    if (NOT LAPACKE_LIB)
        message(FATAL_ERROR "lapacke library not found; install liblapacke-dev or enable MKL")
    endif ()
endif ()

if (APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
endif ()

find_package(OpenMP REQUIRED)

add_subdirectory(third_party/eigen)

set(PLASMA_SOURCE
        control/control.cpp
        control/descriptor.c
        control/context.c
        control/auxiliary.c
        control/allocate.c
        control/async.c
        control/plasmaos.c
        control/plasma_threadsetting.c
        quark/quark.c
        quark/icl_hash.c
        quark/icl_list.c
        quark/quarkos.c
        core_blas/global.c
        core_blas/primes.c
        core_blas/gkkleader.c
        core_blas/descriptor.c
        core_blas-qwrapper/qwrapper_alloc.c

        control/pdtile.c
        control/pdtile_dist.cpp
        core_blas/core_dpotrf.c
        core_blas/core_dsyrk.c
        core_blas/core_dtrsm.c
        core_blas/core_dgemm.c
        core_blas/core_dlacpy.c
        core_blas/core_dgetrip.c
        core_blas/core_dswpab.c
        core_blas/core_dshift.c
        core_blas/core_dplgsy.c
        core_blas-qwrapper/qwrapper_dlacpy.c
        core_blas-qwrapper/qwrapper_dtrsm.c
        core_blas-qwrapper/qwrapper_dsyrk.c
        core_blas-qwrapper/qwrapper_dpotrf.c
        core_blas-qwrapper/qwrapper_dgemm.c
        core_blas-qwrapper/qwrapper_dgetrip.c
        core_blas-qwrapper/qwrapper_dshift.c
        core_blas-qwrapper/qwrapper_dplgsy.c

        compute/dpotrf.c
        compute/pdpotrf.c
        compute/pdpack.c
        compute/pdgetmi2.c
        compute/pdshift.c
        compute/dgecfi.c
        compute/dgecfi2.c
        compute/dlacpy.c
        compute/dplgsy.c
        compute/pdlacpy.c
        compute/pdplgsy.c

        #  POTRI
        #  core_blas/core_dtrtri.c
        #  core_blas-qwrapper/qwrapper_dtrtri.c
        #  core_blas-qwrapper/qwrapper_dlauum.c
        #  core_blas-qwrapper/qwrapper_dtrmm.c
        #  compute/dpotri.c
        #  compute/pdtrtri.c
        #  compute/pdlauum.c

        core_blas-qwrapper/qwrapper_dtrmm.c
        core_blas-qwrapper/qwrapper_dpltmg.c
        core_blas/core_dpltmg.c
        core_blas/core_dplrnt.c
        compute/dtrsm.c
        compute/dtrmm.c
        compute/dgemm.c
        compute/dplrnt.c
        compute/dpltmg.c
        compute/pdtrsm.c
        compute/pdpltmg.c
        compute/pdtrmm.c
)

option(PLASMA_WITH_CUDA "Enable CUDA" ON)
option(PLASMA_WITH_HPCASIA24 "Reproduce results of HPCASIA'24 paper" ON)

if (${PLASMA_WITH_HPCASIA24})
    list(APPEND PLASMA_SOURCE
            compute/dpotrf_hpcasia24.cpp
            compute/mx_emulation.cpp
            compute/dpotrf_mixed_precision.cpp
            compute/pdpotrf_gpu_hpcasia24.cpp
            compute/pdpotrf_gpu_mixed_precision.cpp
            control/event.cpp
    )
endif ()

if (${PLASMA_WITH_CUDA})
    # 80-real/90-real: native SASS for sm_80 (A100) and sm_90 (Hopper/H100),
    # the architectures this has actually been built and run against.
    # 90-virtual: PTX for compute_90 embedded alongside them, which the
    # driver JIT-compiles on any GPU with compute capability >= 9.0 that
    # doesn't have native SASS here -- Blackwell (sm_100/120) and beyond --
    # untested so far, but still runs (via JIT on first launch) rather than
    # hard-failing, without needing a matching -real entry added here.
    set(CMAKE_CUDA_ARCHITECTURES "80-real;90-real;90-virtual" CACHE STRING "CUDA architectures to build for" FORCE)
    find_package(CUDAToolkit REQUIRED)
else ()
    set(CUDAToolkit_FOUND OFF)
endif ()
if (${CUDAToolkit_FOUND})
    enable_language(CUDA)
    list(APPEND PLASMA_SOURCE
            compute/pdpotrf_gpu.c
            compute/dpotrf_gpu.c
            compute/dpotrf_gpu_dist.cpp
            compute/pdpotrf_gpu_dist.cpp
            compute/dpotrf_gpu_dist_mixed_precision.cpp
            compute/pdpotrf_gpu_dist_mixed_precision.cpp
            compute/castType.cu
    )
    add_subdirectory(mixed_kernels)
endif ()

add_library(plasma STATIC ${PLASMA_SOURCE})
target_include_directories(plasma PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/quark>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/control>
)

target_compile_definitions(plasma PRIVATE $<$<COMPILE_LANGUAGE:C>:ADD_>)
if (${CUDAToolkit_FOUND})
    target_compile_definitions(plasma PUBLIC $<$<COMPILE_LANGUAGE:C>:PLASMA_WITH_CUDA>)
    target_compile_definitions(plasma PUBLIC $<$<COMPILE_LANGUAGE:CXX>:PLASMA_WITH_CUDA>)
    target_link_libraries(plasma PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver mixed_kernels)
endif ()

if (${MKL_FOUND})
    target_link_libraries(plasma PUBLIC MKL::MKL)
    target_compile_definitions(plasma PUBLIC $<$<COMPILE_LANGUAGE:C>:PLASMA_WITH_MKL>)
    target_compile_definitions(plasma PUBLIC $<$<COMPILE_LANGUAGE:CXX>:PLASMA_WITH_MKL>)
else ()
    target_link_libraries(plasma PUBLIC ${LAPACKE_LIB} LAPACK::LAPACK BLAS::BLAS)
endif ()

if (${PLASMA_WITH_HPCASIA24})
    target_compile_definitions(plasma PUBLIC $<$<COMPILE_LANGUAGE:C>:PLASMA_WITH_HPCASIA24>)
    target_compile_definitions(plasma PUBLIC $<$<COMPILE_LANGUAGE:CXX>:PLASMA_WITH_HPCASIA24>)
    target_link_libraries(plasma PUBLIC
            Eigen3::Eigen
    )
endif ()

target_link_libraries(plasma PUBLIC OpenMP::OpenMP_C)
target_link_libraries(plasma PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(plasma PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
)

add_executable(example_dpotrf examples/example_dpotrf.c)
target_link_libraries(example_dpotrf PRIVATE plasma)
target_compile_options(example_dpotrf PRIVATE $<$<COMPILE_LANGUAGE:C>:-O3>)

add_executable(example_dpotrf_gpu examples/example_dpotrf_gpu.cpp)
target_link_libraries(example_dpotrf_gpu PRIVATE plasma)
target_compile_options(example_dpotrf_gpu PRIVATE $<$<COMPILE_LANGUAGE:C>:-O3>)
