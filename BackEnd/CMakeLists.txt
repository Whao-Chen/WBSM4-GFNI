cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(WBSM4BackEnd LANGUAGES C)
endif()
include(${CMAKE_CURRENT_LIST_DIR}/options.cmake)
add_library(WBMatrix STATIC EXCLUDE_FROM_ALL
  WBMatrix/src/WBMatrix/WBMatrix.c WBMatrix/src/WBMatrix/random.c)
target_include_directories(WBMatrix PUBLIC WBMatrix/include)
wbs_compile_options(WBMatrix wide)
add_library(WBMatrixGFNI STATIC EXCLUDE_FROM_ALL
  WBMatrix-GFNI/src/WBMatrix/WBMatrix.c WBMatrix-GFNI/src/WBMatrix/random.c
  WBMatrix-GFNI/src/WBMatrix/matrix_mul.c WBMatrix-GFNI/src/WBMatrix/matrix_vector.c
  WBMatrix-GFNI/src/WBMatrix/affine.c WBMatrix-GFNI/src/WBMatrix/hbup.c
  WBMatrix-GFNI/src/WBMatrix/inverse_m8.c)
target_include_directories(WBMatrixGFNI PUBLIC WBMatrix-GFNI/include)
target_include_directories(WBMatrixGFNI PRIVATE WBMatrix-GFNI/src/WBMatrix)
wbs_compile_options(WBMatrixGFNI wide)
file(GLOB COMPAT_SOURCES CONFIGURE_DEPENDS WBMatrix-GFNI/compat/src/*.c)
foreach(profile xl bai wsise wbdl legacy self_equivalence)
  add_library(WBMatrix_${profile} STATIC EXCLUDE_FROM_ALL
    WBMatrix/reference/${profile}/src/WBMatrix.c
    WBMatrix/reference/${profile}/src/random.c)
  target_include_directories(WBMatrix_${profile} PUBLIC WBMatrix/reference/${profile}/include)
  if(profile STREQUAL "self_equivalence")
    wbs_compile_options(WBMatrix_${profile} avx2)
  else()
    wbs_compile_options(WBMatrix_${profile} wide)
  endif()
  add_library(WBMatrixGFNI_${profile} STATIC EXCLUDE_FROM_ALL ${COMPAT_SOURCES}
    WBMatrix/reference/${profile}/src/WBMatrix.c
    WBMatrix/reference/${profile}/src/random.c)
  target_include_directories(WBMatrixGFNI_${profile} PUBLIC
    WBMatrix/reference/legacy/include WBMatrix-GFNI/compat/include
    WBMatrix/reference/${profile}/include)
  wbs_compile_options(WBMatrixGFNI_${profile} wide)
endforeach()
add_library(WBMatrix_self_equivalence_native STATIC EXCLUDE_FROM_ALL
  WBMatrix/reference/self_equivalence/src/WBMatrix.c
  WBMatrix/reference/self_equivalence/src/random.c)
target_include_directories(WBMatrix_self_equivalence_native PUBLIC WBMatrix/reference/self_equivalence/include)
wbs_compile_options(WBMatrix_self_equivalence_native native)
