#pragma once

#define CMAKE_UTILS_DO_PRAGMA(X) _Pragma(#X)

#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
    #define IGNORE_WARNING_PUSH() CMAKE_UTILS_DO_PRAGMA(clang diagnostic push)
    #define IGNORE_WARNING_POP()  CMAKE_UTILS_DO_PRAGMA(clang diagnostic pop)
    #define IGNORE_WARNING(WARN)  CMAKE_UTILS_DO_PRAGMA(clang diagnostic ignored WARN)
#elif defined(__GNUC__)
    #define IGNORE_WARNING_PUSH() CMAKE_UTILS_DO_PRAGMA(GCC diagnostic push)
    #define IGNORE_WARNING_POP()  CMAKE_UTILS_DO_PRAGMA(GCC diagnostic pop)
    #define IGNORE_WARNING(WARN)  CMAKE_UTILS_DO_PRAGMA(GCC diagnostic ignored WARN)
#elif defined(_MSC_VER)
    #define IGNORE_WARNING_PUSH() __pragma(warning(push))
    #define IGNORE_WARNING_POP()  __pragma(warning(pop))
    #define IGNORE_WARNING(WARN)  __pragma(warning(disable : WARN))
#else
    #define IGNORE_WARNING_PUSH()
    #define IGNORE_WARNING_POP()
    #define IGNORE_WARNING(WARN)
#endif

/* Backward-compatible aliases */
#define __START_IGNORING_WARNINGS__ IGNORE_WARNING_PUSH()
#define __STOP_IGNORING_WARNINGS__  IGNORE_WARNING_POP()
#define __IGNORE_WARNING__(WARN)    IGNORE_WARNING(WARN)
