// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   dynamic_library_loader.h
 * @date   14 January 2025
 * @brief  Wrapper for loading dynamic libraries on multiple operating systems
 * @see    https://github.com/nntrainer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __DYNAMIC_LIBRARY_LOADER__
#define __DYNAMIC_LIBRARY_LOADER__

#include <string>

#ifdef _WIN32
#include "windows.h"

// This flags are not used on windows. Defining those symbols for windows make
// possible using the same external interface for loadLibrary function
#define RTLD_LAZY 0
#define RTLD_NOW 0
#define RTLD_BINDING_MASK 0
#define RTLD_NOLOAD 0
#define RTLD_DEEPBIND 0
#define RTLD_GLOBAL 0
#define RTLD_LOCAL 0
#define RTLD_NODELETE 0

#else
#include <dlfcn.h>
#endif

namespace nntrainer {

/**
 * @brief DynamicLibraryLoader wrap process of loading dynamic libraries for
 * multiple operating system
 *
 */
class DynamicLibraryLoader {
public:
  static void *loadLibrary(const char *path, [[maybe_unused]] const int flag) {
    clearLastError();
#if defined(_WIN32)
    return LoadLibraryA(path);
#else
    return dlopen(path, flag);
#endif
  }

  static int freeLibrary(void *handle) {
#if defined(_WIN32)
    return FreeLibrary((HMODULE)handle);
#else
    return dlclose(handle);
#endif
  }

  static const char *getLastError() {
#if defined(_WIN32)
    static thread_local std::string error_message;
    const auto error = GetLastError();
    if (error == ERROR_SUCCESS) {
      error_message.clear();
      return nullptr;
    }

    error_message = "Windows error " + std::to_string(error);
    return error_message.c_str();
#else
    return dlerror();
#endif
  }

  static std::string getLastErrorString() {
    const char *error = getLastError();
    return error == nullptr ? std::string() : std::string(error);
  }

  static void *loadSymbol(void *handle, const char *symbol_name) {
    clearLastError();
#if defined(_WIN32)
    return reinterpret_cast<void *>(
      GetProcAddress((HMODULE)handle, symbol_name));
#else
    return dlsym(handle, symbol_name);
#endif
  }

private:
  static void clearLastError() noexcept {
#if defined(_WIN32)
    SetLastError(ERROR_SUCCESS);
#else
    (void)dlerror();
#endif
  }
};

} // namespace nntrainer

#endif // __DYNAMIC_LIBRARY_LOADER__
