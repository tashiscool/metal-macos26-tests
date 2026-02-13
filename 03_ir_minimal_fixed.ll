; Minimal LLVM IR kernel — WORKS on macOS 26 with proper SDK Version metadata.
; This matches the metadata that the MSL compiler (xcrun metal) generates automatically.

source_filename = "test_kernel"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1

define void @test_kernel(
    i8 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %0,
    i8 addrspace(3)* nocapture noundef readnone "air-buffer-no-alias" %1,
    <3 x i32> noundef %2,
    i16 noundef %3,
    i16 noundef %4
) local_unnamed_addr #0 {
  tail call void @air.wg.barrier(i32 2, i32 1) #3
  ret void
}

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #3 = { convergent nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!20, !21, !22}
!llvm.ident = !{!23}
!air.version = !{!24}
!air.language_version = !{!25}
!air.source_file_name = !{!26}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (i8 addrspace(1)*, i8 addrspace(3)*, <3 x i32>, i16, i16)* @test_kernel, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf", !"air.arg_unused"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"tg_base", !"air.arg_unused"}
!14 = !{i32 2, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
!15 = !{i32 3, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sidx"}
!16 = !{i32 4, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane_id"}
!20 = !{!"air.compile.denorms_disable"}
!21 = !{!"air.compile.fast_math_enable"}
!22 = !{!"air.compile.framebuffer_fetch_enable"}
!23 = !{!"Apple metal version 32023.850 (metalfe-32023.850.10)"}
!24 = !{i32 2, i32 8, i32 0}
!25 = !{!"Metal", i32 4, i32 0, i32 0}
!26 = !{!"test_kernel.metal"}
