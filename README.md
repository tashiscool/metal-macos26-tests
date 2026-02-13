# macOS 26 Metal GPU Compiler Compatibility Tests

Reproduction cases for the AGX GPU compiler crash (`XPC_ERROR_CONNECTION_INTERRUPTED`)
when loading metallibs compiled from handwritten LLVM IR on macOS 26 (Tahoe).

**For the full writeup, see [FINDINGS.md](FINDINGS.md).**

## Quick Start

```bash
bash run_tests.sh
```

Expected output on macOS 26:
```
MSL baseline ... PASS
IR without SDK Version ... EXPECTED FAIL (GPU compiler crash without SDK Version)
IR with SDK Version ... PASS
```

## Root Cause

macOS 26's AGX GPU compiler requires `SDK Version` metadata in the LLVM IR module flags.
Without it, `makeComputePipelineState()` crashes the XPC compilation service.

The MSL compiler (`xcrun metal -c`) adds this automatically. Handwritten IR
(via MetalASM or `xcrun metal -x ir`) does not — you must add it yourself:

```llvm
!llvm.module.flags = !{!0, ...}
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
```

## Test Files

| File | Description | macOS 26 Result |
|------|-------------|-----------------|
| `01_msl_baseline.metal` | MSL kernel (compiler adds metadata) | PASS |
| `02_ir_minimal_broken.ll` | LLVM IR without SDK Version metadata | CRASH |
| `03_ir_minimal_fixed.ll` | LLVM IR with SDK Version metadata | PASS |
| `run_tests.sh` | Automated test runner | — |
| `FINDINGS.md` | Full investigation writeup | — |

## Environment

- macOS 26.2 (Tahoe), Build 25C56
- Apple M4 Pro, 48 GB
- Xcode 26.2 (Build 17C52)
- Metal compiler: Apple metal version 32023.850 (metalfe-32023.850.10)
