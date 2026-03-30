arch = host_machine.cpu_family()

if arch == 'x86_64' or arch == 'x86'
  executable(
    'bench_ggml_kernels',
    'bench_ggml_kernels.cpp',
    dependencies: [nntrainer_dep],
    install: false
  )
else
  message('Skipping perf/bench_ggml_kernels on non-x86 host')
endif
