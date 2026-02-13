; Minimal LLVM IR kernel — CRASHES on macOS 26 because it's missing SDK Version metadata.
; The GPU compiler (AGX) needs this metadata or makeComputePipelineState() fails with
; XPC_ERROR_CONNECTION_INTERRUPTED.

source_filename = "test_kernel"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare void @air.wg.barrier(i32, i32) #1

define void @test_kernel(
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf,
    i8 addrspace(3)* noundef %tg_base,
    <3 x i32> noundef %gid,
    i16 noundef %sidx_i16,
    i16 noundef %lane_id_i16
) local_unnamed_addr #0 {
entry:
  call void @air.wg.barrier(i32 2, i32 1)
  ret void
}

attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" "no-builtins" "no-trapping-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }

; MISSING: SDK Version, air.max_*, air.compile_options
!air.kernel = !{!0}
!llvm.module.flags = !{!8}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{void (i8 addrspace(1)*, i8 addrspace(3)*, <3 x i32>, i16, i16)* @test_kernel, !1, !2}
!1 = !{}
!2 = !{!30, !40, !41, !42, !43}
!30 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1}
!40 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1}
!41 = !{i32 2, !"air.threadgroup_position_in_grid"}
!42 = !{i32 3, !"air.simdgroup_index_in_threadgroup"}
!43 = !{i32 4, !"air.thread_index_in_simdgroup"}

!8 = !{i32 1, !"wchar_size", i32 4}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
