function(wbs_compile_options target level)
  target_compile_features(${target} PRIVATE c_std_99)
  target_compile_options(${target} PRIVATE -O2 -fcommon)
  if(level STREQUAL "avx2")
    target_compile_options(${target} PRIVATE -m64 -mavx2)
  elseif(level STREQUAL "native")
    target_compile_options(${target} PRIVATE -march=native -m64 -mavx2)
  else()
    target_compile_options(${target} PRIVATE -march=native -m64 -mavx2 -mgfni -mbmi2
      -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vpopcntdq
      -mavx512vbmi -mavx512bitalg -mvpclmulqdq)
  endif()
endfunction()
