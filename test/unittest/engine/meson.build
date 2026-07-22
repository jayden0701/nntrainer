if get_option('default_library') != 'static' and not meson.is_cross_build()
  fake_context_plugins = []

  foreach plugin_id : [1, 2, 3, 4, 5]
    fake_plugin_args = [
      '-DNNTR_FAKE_CONTEXT_ID=@0@'.format(plugin_id),
    ]
    if plugin_id == 5
      fake_plugin_args += ['-DNNTR_FAKE_CONTEXT_NAME_ID=1']
    endif

    if host_machine.system() == 'windows'
      fake_context_plugin = shared_module(
        'nntrainer_fake_context_plugin_@0@'.format(plugin_id),
        'fake_context_plugin.cpp',
        cpp_args: fake_plugin_args,
        dependencies: nntrainer_dep,
        vs_module_defs: 'fake_context_plugin.def',
        install: false
      )
    else
      fake_context_plugin = shared_module(
        'nntrainer_fake_context_plugin_@0@'.format(plugin_id),
        'fake_context_plugin.cpp',
        cpp_args: fake_plugin_args,
        dependencies: nntrainer_dep,
        install: false
      )
    endif
    fake_context_plugins += [fake_context_plugin]
  endforeach

  engine_context_plugin_test = executable(
    'unittest_engine_context_plugin',
    'unittest_engine_context_plugin.cpp',
    dependencies: [nntrainer_dep, thread_dep],
    install: false
  )

  engine_context_plugin_test_env = environment()
  engine_context_plugin_test_env.set('NNTR_NUM_THREADS', '1')
  foreach plugin_id : [1, 2, 3, 4, 5]
    engine_context_plugin_test_env.set(
      'NNTR_FAKE_CONTEXT_PLUGIN_@0@'.format(plugin_id),
      fake_context_plugins[plugin_id - 1].full_path()
    )
  endforeach

  test(
    'unittest_engine_context_plugin',
    engine_context_plugin_test,
    env: engine_context_plugin_test_env,
    depends: fake_context_plugins,
    timeout: test_timeout,
    suite: 'unittests'
  )
endif
