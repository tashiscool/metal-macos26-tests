source_filename = "monolithic_gemm"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%event_t = type opaque

  declare void @air.wg.barrier(i32, i32) #1
  declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2
  declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2


declare float @air.simd_shuffle_xor.f32(float, i32) #1
declare float @llvm.exp2.f32(float) #1
declare float @llvm.log2.f32(float) #1
  declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half>, <64 x half>, <64 x float>) local_unnamed_addr #1
  declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float>, <64 x half>, <64 x float>) local_unnamed_addr #1

define void @attention(
    i8 addrspace(1)* noundef "air-buffer-no-alias" %Q_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %K_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %V_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %O_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %L_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %D_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %dO_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %dV_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %dK_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %dQ_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %mask_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %bias_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %K_scale_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %V_scale_base,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %bp_raw,
    i8 addrspace(3)* noundef %tg_base,
    <3 x i32> noundef %gid,
    i16 noundef %sidx_i16,
    i16 noundef %lane_id_i16
) local_unnamed_addr #0 {
entry:
  ret void
}



attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" "min-legal-vector-width"="96" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { convergent nounwind willreturn }
attributes #4 = { nounwind }

!air.kernel = !{!0}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13, !14}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!19}
!air.version = !{!20}
!air.language_version = !{!21}
!air.source_file_name = !{!22}

!0 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(3)*, <3 x i32>, i16, i16)* @attention, !1, !2}
!1 = !{}
!2 = !{!30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !44, !45, !47, !48, !46, !40, !41, !42, !43}
!30 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"Q"}
!31 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"K"}
!32 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"V"}
!33 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"O"}
!34 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"L"}
!35 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"D_buf"}
!36 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dO"}
!37 = !{i32 7, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dV"}
!38 = !{i32 8, !"air.buffer", !"air.location_index", i32 8, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dK"}
!39 = !{i32 9, !"air.buffer", !"air.location_index", i32 9, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dQ"}
!44 = !{i32 10, !"air.buffer", !"air.location_index", i32 10, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"mask"}
!45 = !{i32 11, !"air.buffer", !"air.location_index", i32 11, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"attn_bias"}
!47 = !{i32 12, !"air.buffer", !"air.location_index", i32 20, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"K_scale"}
!48 = !{i32 13, !"air.buffer", !"air.location_index", i32 21, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"V_scale"}
!46 = !{i32 14, !"air.buffer", !"air.location_index", i32 30, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"batch_params"}
!40 = !{i32 15, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"tg_mem"}
!41 = !{i32 16, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
!42 = !{i32 17, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sidx"}
!43 = !{i32 18, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane_id"}

!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"air.max_device_buffers", i32 31}
!10 = !{i32 7, !"air.max_constant_buffers", i32 31}
!11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!12 = !{i32 7, !"air.max_textures", i32 128}
!13 = !{i32 7, !"air.max_read_write_textures", i32 8}
!14 = !{i32 7, !"air.max_samplers", i32 16}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"MetalASM (monolithic attention)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"monolithic_attention.ll"}
