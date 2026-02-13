# macOS 26 (Tahoe) Metal LLVM IR Compatibility — Full Findings

> **TL;DR**: macOS 26's AGX GPU compiler crashes (`XPC_ERROR_CONNECTION_INTERRUPTED`) when
> compiling metallibs from handwritten LLVM IR that is missing `SDK Version` module metadata.
> Adding one line of metadata fixes minimal kernels. Complex kernels (e.g. flash attention
> with async copy intrinsics) still crash even with full metadata — the root cause there
> remains under investigation.

## Environment

| Component | Version |
|-----------|---------|
| macOS | 26.2 (Tahoe), Build 25C56 |
| Hardware | MacBook Pro (Mac16,7), Apple M4 Pro, 48 GB |
| Xcode | 26.2 (Build 17C52) |
| Metal compiler | Apple metal version 32023.850 (metalfe-32023.850.10) |
| Metal AIR assembler | Apple LLVM version 32023.850.10 (`xcrun air-as`) |
| AIR target triple | `air64_v28-apple-macosx26.0.0` |

## Background

Metal shaders can be compiled from two sources:

1. **MSL (Metal Shading Language)** — `.metal` files compiled by `xcrun metal -c`. The compiler
   automatically generates all required AIR metadata (SDK Version, resource limits, compile
   options, argument descriptors). This path **always works** on macOS 26.

2. **LLVM IR** — `.ll` files containing handwritten AIR-compatible LLVM IR. These can be
   assembled via `xcrun metal -x ir -c`, `xcrun air-as`, or third-party tools like MetalASM.
   The author must manually supply ALL metadata. Missing or incorrect metadata causes the GPU
   compiler to crash on macOS 26.

Projects that generate LLVM IR directly (to bypass MSL limitations or for performance) are
affected. This includes [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
v0.5.0+ which uses a MetalASM IR backend.

## The Crash

When `MTLDevice.makeComputePipelineState(function:)` is called with a metallib compiled from
IR that lacks required metadata, the AGX GPU compiler's XPC service crashes:

```
Error Domain=AGXMetalG16X Code=2
"Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED.
 This error occurred after multiple retries."
```

This is a **hard crash** of the GPU compiler service, not a validation error. The metallib
loads fine (`makeLibrary` succeeds, `makeFunction` succeeds) — the crash happens only when
the GPU compiler tries to compile the function to machine code.

On macOS 15 and earlier, the same metallibs work without issue. The stricter metadata
validation is new in macOS 26.

## What We Discovered

### Finding 1: `SDK Version` metadata is REQUIRED

The single most critical piece of missing metadata is:

```llvm
!llvm.module.flags = !{!0, ...}
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
```

This tells the GPU compiler which SDK the code was compiled against. Without it, the XPC
service crashes. The MSL compiler adds this automatically; handwritten IR does not.

**Proof**: `02_ir_minimal_broken.ll` (no SDK Version) crashes. `03_ir_minimal_fixed.ll`
(with SDK Version) works. The only difference is the metadata.

### Finding 2: Full metadata set needed for production kernels

Minimal kernels work with just `SDK Version`, but production kernels need the full set:

```llvm
; Module flags (all required on macOS 26)
!llvm.module.flags = !{!50, !8, !9, !10, !11, !12, !13, !14}
!50 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
!8  = !{i32 1, !"wchar_size", i32 4}
!9  = !{i32 7, !"air.max_device_buffers", i32 31}
!10 = !{i32 7, !"air.max_constant_buffers", i32 31}
!11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!12 = !{i32 7, !"air.max_textures", i32 128}
!13 = !{i32 7, !"air.max_read_write_textures", i32 8}
!14 = !{i32 7, !"air.max_samplers", i32 16}

; Compile options
!air.compile_options = !{!15, !16, !17}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}

; Version info
!air.version = !{i32 2, i32 8, i32 0}              ; AIR 2.8.0
!air.language_version = !{!"Metal", i32 4, i32 0, i32 0}  ; Metal 4.0.0
!llvm.ident = !{!"Apple metal version 32023.850 (metalfe-32023.850.10)"}
!air.source_file_name = !{!"my_kernel.metal"}
```

### Finding 3: Kernel argument metadata must include type names

Every kernel argument (buffers, threadgroup memory, and system values) needs complete
metadata including `air.arg_type_name` and `air.arg_name`:

```llvm
; Buffer argument — COMPLETE metadata required
!30 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1,
        !"air.read_write", !"air.address_space", i32 1,
        !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1,
        !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf"}

; System value — also needs type name
!41 = !{i32 16, !"air.threadgroup_position_in_grid",
        !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
```

Compare with the "broken" test case which omits `air.arg_type_name` — this works on macOS 15
but crashes on macOS 26.

### Finding 4: Complex intrinsics may have additional issues

Even with full correct metadata, **complex kernels using `@air.simdgroup_async_copy_2d`
intrinsics still crash** on macOS 26. This was tested with:

- The full flash attention forward kernel (~3000 lines of IR)
- All metadata present and correct (SDK Version, air.max_*, compile options, arg types)
- Compiled via MetalASM, `xcrun metal -x ir -c`, and `xcrun air-as` — all three crash

The crash is the same `XPC_ERROR_CONNECTION_INTERRUPTED`. The backward attention kernel
(which does NOT use async copy) works fine with the metadata fix.

**Status**: This is an open issue. The async copy intrinsic signature matches what the MSL
compiler generates, but something about the surrounding IR may be triggering a GPU compiler
bug. Workaround: replace async copy with synchronous threadgroup loads (all threads
cooperatively copy data, followed by a barrier).

### Finding 5: Assembly tools behave differently

| Tool | Command | Result |
|------|---------|--------|
| MSL compiler | `xcrun metal -c foo.metal` | Always works, adds all metadata |
| IR via metal | `xcrun metal -x ir -c foo.ll` | Compiles to .air, but may modify IR |
| AIR assembler | `xcrun air-as foo.ll -o foo.air` | Direct IR→bitcode, no modification |
| MetalASM | Swift library, IR→metallib directly | Bundles IR into metallib container |

For handwritten IR, `air-as` + `metallib` is the most reliable pipeline. The `metal -x ir -c`
frontend may rewrite or reinterpret IR in unexpected ways.

### Finding 6: The target triple matters

```llvm
target triple = "air64_v28-apple-macosx26.0.0"
```

- `air64` — 64-bit AIR
- `v28` — AIR version 2.8 (must match `!air.version = !{i32 2, i32 8, i32 0}`)
- `macosx26.0.0` — minimum deployment target

Using an older target (e.g. `macosx15.0.0`) with SDK Version 26 is untested. The safest
approach is to match the target triple to the running OS version.

## Checklist for Handwritten Metal LLVM IR on macOS 26

If you write LLVM IR targeting Apple's AIR (for Metal GPU kernels), ensure you have ALL of:

- [ ] `target triple = "air64_v28-apple-macosx26.0.0"`
- [ ] `!{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}` in `!llvm.module.flags`
- [ ] `air.max_device_buffers`, `air.max_constant_buffers`, `air.max_threadgroup_buffers`,
      `air.max_textures`, `air.max_read_write_textures`, `air.max_samplers` in module flags
- [ ] `!air.compile_options` with denorms_disable, fast_math_enable, framebuffer_fetch_enable
- [ ] `!air.version = !{i32 2, i32 8, i32 0}`
- [ ] `!air.language_version = !{!"Metal", i32 4, i32 0, i32 0}`
- [ ] `!llvm.ident` with a plausible Metal compiler version string
- [ ] `!air.source_file_name` with any `.metal` filename
- [ ] All kernel arguments have `air.arg_type_name` and `air.arg_name` in their metadata
- [ ] Buffer arguments have `air.arg_type_size` and `air.arg_type_align_size`
- [ ] Function attributes include `convergent mustprogress nounwind willreturn`

## Reproducing

```bash
git clone https://github.com/tashiscool/metal-macos26-tests
cd metal-macos26-tests
bash run_tests.sh
```

Expected output on macOS 26:
```
MSL baseline ... PASS
IR without SDK Version ... EXPECTED FAIL (GPU compiler crash without SDK Version)
IR with SDK Version ... PASS
```

Expected output on macOS 15 or earlier:
```
MSL baseline ... PASS
IR without SDK Version ... PASS
IR with SDK Version ... PASS
```

## Affected Projects

- **[metal-flash-attention](https://github.com/philipturner/metal-flash-attention)** v0.5.0+ —
  Uses MetalASM IR backend. Forward attention crashes on macOS 26 (async copy + metadata).
  Backward attention works with the SDK Version metadata fix.

- **Any project** using handwritten LLVM IR for Metal compute kernels — if you generate `.ll`
  files and assemble them to `.metallib`, you need the metadata listed above.

## Timeline

- **macOS 15 and earlier**: Handwritten IR works without SDK Version metadata
- **macOS 26 beta (Tahoe)**: GPU compiler crashes without SDK Version metadata
- **2025-02-12**: Root cause identified; metadata fix confirmed for simple kernels;
  async copy intrinsic issue remains open for complex kernels

## References

- Apple Metal Shading Language Specification (AIR chapter)
- `xcrun metal --help` / `xcrun air-as --help` / `xcrun metallib --help`
- LLVM IR reference: https://llvm.org/docs/LangRef.html
- MetalASM: https://github.com/philipturner/metal-flash-attention (embedded in v0.5.0+)
