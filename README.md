# macOS 26 Metal GPU Compiler Compatibility Tests

Reproduction cases for the AGX GPU compiler crash (`XPC_ERROR_CONNECTION_INTERRUPTED`)
when loading metallibs compiled from handwritten LLVM IR on macOS 26 (Tahoe).

## Root Cause

macOS 26's AGX GPU compiler requires `SDK Version` metadata in the LLVM IR module flags.
Without it, `makeComputePipelineState()` crashes the XPC compilation service.

The MSL compiler (`xcrun metal -c`) adds this automatically. Handwritten IR
(via MetalASM or `xcrun metal -x ir`) does not — you must add it yourself.

## Required metadata (minimum)

```llvm
!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
```

Also needed:
```llvm
!air.compile_options = !{!20, !21, !22}
!20 = !{!"air.compile.denorms_disable"}
!21 = !{!"air.compile.fast_math_enable"}
!22 = !{!"air.compile.framebuffer_fetch_enable"}
```

## Test files

- `01_msl_baseline.metal` - MSL kernel (always works)
- `02_ir_minimal_broken.ll` - Minimal IR without SDK Version (CRASHES)
- `03_ir_minimal_fixed.ll` - Minimal IR with SDK Version (WORKS)
- `04_ir_with_barrier.ll` - IR with threadgroup barrier (tests intrinsics)
- `run_tests.sh` - Automated test runner

## Environment

- macOS 26.2 (Tahoe) beta
- Apple M4 Pro
- Xcode 17.3 beta
- Metal compiler: Apple metal version 32023.850
