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
  %sidx = zext i16 %sidx_i16 to i32
  %lane_id = zext i16 %lane_id_i16 to i32
  %gid_x = extractelement <3 x i32> %gid, i64 0
  %gid_y = extractelement <3 x i32> %gid, i64 1
  ; === Load batch params ===
  %bp_ptr = bitcast i8 addrspace(1)* %bp_raw to i32 addrspace(1)*
  ; bp[0] = numHeads, bp[1] = kvRepeatFactor
  %numHeads = load i32, i32 addrspace(1)* %bp_ptr
  %bp_kvr_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 1
  %kvRepeatFactor = load i32, i32 addrspace(1)* %bp_kvr_ptr

  ; Head indices: batch_head_idx = gid.y
  %batch_head_idx = bitcast i32 %gid_y to i32
  %kv_head_idx = udiv i32 %batch_head_idx, %kvRepeatFactor
  ; Decompose into batch and head for bias strides
  %batch_idx = udiv i32 %batch_head_idx, %numHeads
  %head_idx = urem i32 %batch_head_idx, %numHeads

  ; Load K/V dequantization scales (float, indexed by batch_head_idx)
  %K_scale_fptr = bitcast i8 addrspace(1)* %K_scale_base to float addrspace(1)*
  %K_scale_ptr = getelementptr float, float addrspace(1)* %K_scale_fptr, i32 %batch_head_idx
  %K_scale = load float, float addrspace(1)* %K_scale_ptr
  %V_scale_fptr = bitcast i8 addrspace(1)* %V_scale_base to float addrspace(1)*
  %V_scale_ptr = getelementptr float, float addrspace(1)* %V_scale_fptr, i32 %batch_head_idx
  %V_scale = load float, float addrspace(1)* %V_scale_ptr

  ; Load per-operand strides from bp[2..13]
  %stride_Q_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 2
  %stride_Q = load i32, i32 addrspace(1)* %stride_Q_ptr
  %stride_K_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 3
  %stride_K = load i32, i32 addrspace(1)* %stride_K_ptr
  %stride_V_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 4
  %stride_V = load i32, i32 addrspace(1)* %stride_V_ptr
  %stride_O_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 5
  %stride_O = load i32, i32 addrspace(1)* %stride_O_ptr
  %stride_L_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 6
  %stride_L = load i32, i32 addrspace(1)* %stride_L_ptr
  %stride_D_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 7
  %stride_D = load i32, i32 addrspace(1)* %stride_D_ptr
  %stride_dO_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 8
  %stride_dO = load i32, i32 addrspace(1)* %stride_dO_ptr
  %stride_dV_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 9
  %stride_dV = load i32, i32 addrspace(1)* %stride_dV_ptr
  %stride_dK_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 10
  %stride_dK = load i32, i32 addrspace(1)* %stride_dK_ptr
  %stride_dQ_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 11
  %stride_dQ = load i32, i32 addrspace(1)* %stride_dQ_ptr
  %causal_off_ptr = getelementptr i32, i32 addrspace(1)* %bp_ptr, i64 12
  %causal_offset = load i32, i32 addrspace(1)* %causal_off_ptr

  %off_Q_elem = mul i32 %batch_head_idx, %stride_Q
  %off_Q_32 = mul i32 %off_Q_elem, 2
  %off_Q = zext i32 %off_Q_32 to i64
  %Q = getelementptr i8, i8 addrspace(1)* %Q_base, i64 %off_Q
  %off_O_elem = mul i32 %batch_head_idx, %stride_O
  %off_O_32 = mul i32 %off_O_elem, 2
  %off_O = zext i32 %off_O_32 to i64
  %O = getelementptr i8, i8 addrspace(1)* %O_base, i64 %off_O
  %off_L_elem = mul i32 %batch_head_idx, %stride_L
  %off_L_32 = mul i32 %off_L_elem, 4
  %off_L = zext i32 %off_L_32 to i64
  %L = getelementptr i8, i8 addrspace(1)* %L_base, i64 %off_L
  %off_D_elem = mul i32 %batch_head_idx, %stride_D
  %off_D_32 = mul i32 %off_D_elem, 4
  %off_D = zext i32 %off_D_32 to i64
  %D = getelementptr i8, i8 addrspace(1)* %D_base, i64 %off_D
  %off_dO_elem = mul i32 %batch_head_idx, %stride_dO
  %off_dO_32 = mul i32 %off_dO_elem, 2
  %off_dO = zext i32 %off_dO_32 to i64
  %dO = getelementptr i8, i8 addrspace(1)* %dO_base, i64 %off_dO
  %off_dQ_elem = mul i32 %batch_head_idx, %stride_dQ
  %off_dQ_32 = mul i32 %off_dQ_elem, 4
  %off_dQ = zext i32 %off_dQ_32 to i64
  %dQ = getelementptr i8, i8 addrspace(1)* %dQ_base, i64 %off_dQ
  %off_K_elem = mul i32 %kv_head_idx, %stride_K
  %off_K_32 = mul i32 %off_K_elem, 2
  %off_K = zext i32 %off_K_32 to i64
  %K = getelementptr i8, i8 addrspace(1)* %K_base, i64 %off_K
  %off_V_elem = mul i32 %kv_head_idx, %stride_V
  %off_V_32 = mul i32 %off_V_elem, 2
  %off_V = zext i32 %off_V_32 to i64
  %V = getelementptr i8, i8 addrspace(1)* %V_base, i64 %off_V
  %off_dV_elem = mul i32 %kv_head_idx, %stride_dV
  %off_dV_32 = mul i32 %off_dV_elem, 4
  %off_dV = zext i32 %off_dV_32 to i64
  %dV = getelementptr i8, i8 addrspace(1)* %dV_base, i64 %off_dV
  %off_dK_elem = mul i32 %kv_head_idx, %stride_dK
  %off_dK_32 = mul i32 %off_dK_elem, 4
  %off_dK = zext i32 %off_dK_32 to i64
  %dK = getelementptr i8, i8 addrspace(1)* %dK_base, i64 %off_dK
  %L_buf = bitcast i8 addrspace(1)* %L to i8 addrspace(1)*
  %D_buf = bitcast i8 addrspace(1)* %D to i8 addrspace(1)*

  ; === Morton order computation ===
  %q = lshr i32 %lane_id, 2
  %m_floor = and i32 %q, 16380
  %h = lshr i32 %lane_id, 1
  %m_in_quad = and i32 %h, 3
  %morton_y = or i32 %m_floor, %m_in_quad
  %n_floor = and i32 %h, 4
  %n_in_quad_s = shl i32 %lane_id, 1
  %n_in_quad = and i32 %n_in_quad_s, 2
  %morton_x = or i32 %n_floor, %n_in_quad

  ; parallelization_group_offset = gid.x * blockP
  %par_group_off = mul i32 %gid_x, 16

  ; Early return if entire group is out of bounds
  %early_oob = icmp uge i32 %par_group_off, 64
  br i1 %early_oob, label %exit, label %valid_group

valid_group:
  ; Gate async copies to simdgroup 0
  %is_sidx0 = icmp eq i32 %sidx, 0

  ; Compute thread offsets within group
  ; oig_y = sidx * 8 + morton_y (parallelization offset within group)
  %oig_y_base = mul i32 %sidx, 8
  %oig_y = add i32 %oig_y_base, %morton_y

  ; unsafe_par_thread_off = par_group_off + oig_y
  %unsafe_par_off = add i32 %par_group_off, %oig_y
  ; clamped_par_off = min(unsafe_par_off, parallelDim - 1)
  %par_dim_m1 = sub i32 64, 1
  %par_cmp = icmp ult i32 %unsafe_par_off, 64
  %clamped_par_off = select i1 %par_cmp, i32 %unsafe_par_off, i32 %par_dim_m1

  ; causal_row = unsafe_par_off + causal_offset
  %causal_row = add i32 %unsafe_par_off, %causal_offset
  ; === Forward setup ===
  %o_init_0 = bitcast <64 x float> zeroinitializer to <64 x float>
  %o_init_1 = bitcast <64 x float> zeroinitializer to <64 x float>
  %o_init_2 = bitcast <64 x float> zeroinitializer to <64 x float>
  %o_init_3 = bitcast <64 x float> zeroinitializer to <64 x float>
  %o_init_4 = bitcast <64 x float> zeroinitializer to <64 x float>
  %o_init_5 = bitcast <64 x float> zeroinitializer to <64 x float>
  %o_init_6 = bitcast <64 x float> zeroinitializer to <64 x float>
  %o_init_7 = bitcast <64 x float> zeroinitializer to <64 x float>
  ; m = -MAX, l = denorm_min
  %m_init = bitcast float 0xFFF0000000000000 to float ; -inf
  %l_init = bitcast float 0x36A0000000000000 to float ; denorm_min
  ; === Prologue: prefetch K[0, d_outer=0] → TG slot A ===
  ; === Sync copy (pre_k_) — all threads cooperative ===
  %pre_k_sc_t0 = mul i32 %sidx, 32
  %pre_k_sc_tid = add i32 %pre_k_sc_t0, %lane_id
  %pre_k_sc_drem = sub i32 64, 0
  %pre_k_sc_dcmp = icmp ult i32 %pre_k_sc_drem, 16
  %pre_k_sc_dsrc = select i1 %pre_k_sc_dcmp, i32 %pre_k_sc_drem, i32 16
  %pre_k_sc_srem = sub i32 64, 0
  %pre_k_sc_scmp = icmp ult i32 %pre_k_sc_srem, 64
  %pre_k_sc_ssrc = select i1 %pre_k_sc_scmp, i32 %pre_k_sc_srem, i32 64
  br label %pre_k_sc_pre

pre_k_sc_pre:
  br label %pre_k_sc_hdr

pre_k_sc_hdr:
  %pre_k_sc_i = phi i32 [%pre_k_sc_tid, %pre_k_sc_pre], [%pre_k_sc_inx, %pre_k_sc_st]
  %pre_k_sc_done = icmp uge i32 %pre_k_sc_i, 1024
  br i1 %pre_k_sc_done, label %pre_k_sc_end, label %pre_k_sc_body

pre_k_sc_body:
  %pre_k_sc_row = lshr i32 %pre_k_sc_i, 4
  %pre_k_sc_col = and i32 %pre_k_sc_i, 15
  %pre_k_sc_rok = icmp ult i32 %pre_k_sc_row, %pre_k_sc_ssrc
  %pre_k_sc_cok = icmp ult i32 %pre_k_sc_col, %pre_k_sc_dsrc
  %pre_k_sc_ib = and i1 %pre_k_sc_rok, %pre_k_sc_cok
  br i1 %pre_k_sc_ib, label %pre_k_sc_ld, label %pre_k_sc_zr

pre_k_sc_ld:
  %pre_k_sc_sr = add i32 0, %pre_k_sc_row
  %pre_k_sc_sa = mul i32 %pre_k_sc_sr, 64
  %pre_k_sc_sc = add i32 0, %pre_k_sc_col
  %pre_k_sc_sad = add i32 %pre_k_sc_sa, %pre_k_sc_sc
  %pre_k_sc_soff = zext i32 %pre_k_sc_sad to i64
  %pre_k_sc_sbyt = mul i64 %pre_k_sc_soff, 2
  %pre_k_sc_sp = getelementptr i8, i8 addrspace(1)* %K, i64 %pre_k_sc_sbyt
  %pre_k_sc_spt = bitcast i8 addrspace(1)* %pre_k_sc_sp to i16 addrspace(1)*
  %pre_k_sc_lv = load i16, i16 addrspace(1)* %pre_k_sc_spt
  br label %pre_k_sc_st

pre_k_sc_zr:
  br label %pre_k_sc_st

pre_k_sc_st:
  %pre_k_sc_val = phi i16 [%pre_k_sc_lv, %pre_k_sc_ld], [0, %pre_k_sc_zr]
  %pre_k_sc_tr = mul i32 %pre_k_sc_row, 16
  %pre_k_sc_ta = add i32 %pre_k_sc_tr, %pre_k_sc_col
  %pre_k_sc_tb = mul i32 %pre_k_sc_ta, 2
  %pre_k_sc_tb64 = zext i32 %pre_k_sc_tb to i64
  %pre_k_sc_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %pre_k_sc_tb64
  %pre_k_sc_tpt = bitcast i8 addrspace(3)* %pre_k_sc_tp to i16 addrspace(3)*
  store i16 %pre_k_sc_val, i16 addrspace(3)* %pre_k_sc_tpt
  %pre_k_sc_inx = add i32 %pre_k_sc_i, 64
  br label %pre_k_sc_hdr

pre_k_sc_end:
  call void @air.wg.barrier(i32 2, i32 1)


  ; === Traversal loop (double-buffered) ===
  br label %before_loop
before_loop:
  br label %loop_header

loop_header:
  %c = phi i32 [0, %before_loop], [%c_next, %loop_latch]
  %m_phi = phi float [%m_init, %before_loop], [%m_updated, %loop_latch]
  %l_phi = phi float [%l_init, %before_loop], [%l_updated, %loop_latch]
  %o_phi_0 = phi <64 x float> [%o_init_0, %before_loop], [%o_acc_0, %loop_latch]
  %o_phi_1 = phi <64 x float> [%o_init_1, %before_loop], [%o_acc_1, %loop_latch]
  %o_phi_2 = phi <64 x float> [%o_init_2, %before_loop], [%o_acc_2, %loop_latch]
  %o_phi_3 = phi <64 x float> [%o_init_3, %before_loop], [%o_acc_3, %loop_latch]
  %o_phi_4 = phi <64 x float> [%o_init_4, %before_loop], [%o_acc_4, %loop_latch]
  %o_phi_5 = phi <64 x float> [%o_init_5, %before_loop], [%o_acc_5, %loop_latch]
  %o_phi_6 = phi <64 x float> [%o_init_6, %before_loop], [%o_acc_6, %loop_latch]
  %o_phi_7 = phi <64 x float> [%o_init_7, %before_loop], [%o_acc_7, %loop_latch]

  %loop_done = icmp uge i32 %c, 64
  br i1 %loop_done, label %cleanup, label %loop_body

loop_body:
  ; === Start V[c, d_outer=0] → TG slot B (no wait) ===
  ; === Sync copy (pv_) — all threads cooperative ===
  %pv_sc_t0 = mul i32 %sidx, 32
  %pv_sc_tid = add i32 %pv_sc_t0, %lane_id
  %pv_sc_drem = sub i32 64, 0
  %pv_sc_dcmp = icmp ult i32 %pv_sc_drem, 16
  %pv_sc_dsrc = select i1 %pv_sc_dcmp, i32 %pv_sc_drem, i32 16
  %pv_sc_soob = icmp uge i32 %c, 64
  %pv_sc_srr = sub i32 64, %c
  %pv_sc_srem = select i1 %pv_sc_soob, i32 0, i32 %pv_sc_srr
  %pv_sc_scmp = icmp ult i32 %pv_sc_srem, 64
  %pv_sc_ssrc = select i1 %pv_sc_scmp, i32 %pv_sc_srem, i32 64
  br label %pv_sc_pre

pv_sc_pre:
  br label %pv_sc_hdr

pv_sc_hdr:
  %pv_sc_i = phi i32 [%pv_sc_tid, %pv_sc_pre], [%pv_sc_inx, %pv_sc_st]
  %pv_sc_done = icmp uge i32 %pv_sc_i, 1024
  br i1 %pv_sc_done, label %pv_sc_end, label %pv_sc_body

pv_sc_body:
  %pv_sc_row = lshr i32 %pv_sc_i, 4
  %pv_sc_col = and i32 %pv_sc_i, 15
  %pv_sc_rok = icmp ult i32 %pv_sc_row, %pv_sc_ssrc
  %pv_sc_cok = icmp ult i32 %pv_sc_col, %pv_sc_dsrc
  %pv_sc_ib = and i1 %pv_sc_rok, %pv_sc_cok
  br i1 %pv_sc_ib, label %pv_sc_ld, label %pv_sc_zr

pv_sc_ld:
  %pv_sc_sr = add i32 %c, %pv_sc_row
  %pv_sc_sa = mul i32 %pv_sc_sr, 64
  %pv_sc_sc = add i32 0, %pv_sc_col
  %pv_sc_sad = add i32 %pv_sc_sa, %pv_sc_sc
  %pv_sc_soff = zext i32 %pv_sc_sad to i64
  %pv_sc_sbyt = mul i64 %pv_sc_soff, 2
  %pv_sc_sp = getelementptr i8, i8 addrspace(1)* %V, i64 %pv_sc_sbyt
  %pv_sc_spt = bitcast i8 addrspace(1)* %pv_sc_sp to i16 addrspace(1)*
  %pv_sc_lv = load i16, i16 addrspace(1)* %pv_sc_spt
  br label %pv_sc_st

pv_sc_zr:
  br label %pv_sc_st

pv_sc_st:
  %pv_sc_val = phi i16 [%pv_sc_lv, %pv_sc_ld], [0, %pv_sc_zr]
  %pv_sc_tr = mul i32 %pv_sc_row, 16
  %pv_sc_ta = add i32 %pv_sc_tr, %pv_sc_col
  %pv_sc_tb = mul i32 %pv_sc_ta, 2
  %pv_sc_tb64 = zext i32 %pv_sc_tb to i64
  %pv_sc_tb64o = add i64 %pv_sc_tb64, 2048
  %pv_sc_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %pv_sc_tb64o
  %pv_sc_tpt = bitcast i8 addrspace(3)* %pv_sc_tp to i16 addrspace(3)*
  store i16 %pv_sc_val, i16 addrspace(3)* %pv_sc_tpt
  %pv_sc_inx = add i32 %pv_sc_i, 64
  br label %pv_sc_hdr

pv_sc_end:
  call void @air.wg.barrier(i32 2, i32 1)

  %s_init_0 = bitcast <64 x float> zeroinitializer to <64 x float>
  %s_init_1 = bitcast <64 x float> zeroinitializer to <64 x float>
  %s_init_2 = bitcast <64 x float> zeroinitializer to <64 x float>
  %s_init_3 = bitcast <64 x float> zeroinitializer to <64 x float>
  %s_init_4 = bitcast <64 x float> zeroinitializer to <64 x float>
  %s_init_5 = bitcast <64 x float> zeroinitializer to <64 x float>
  %s_init_6 = bitcast <64 x float> zeroinitializer to <64 x float>
  %s_init_7 = bitcast <64 x float> zeroinitializer to <64 x float>
  ; === Outer Product Q * K^T → s ===
  %op_0_a0_seq = add i32 %clamped_par_off, 0
  %op_0_a0_head = add i32 %morton_x, 0
  %op_0_a0_addr = mul i32 %op_0_a0_seq, 64
  %op_0_a0_addr2 = add i32 %op_0_a0_addr, %op_0_a0_head
  %op_0_a0_byte = mul i32 %op_0_a0_addr2, 2
  %op_0_a0_byte64 = zext i32 %op_0_a0_byte to i64
  %op_0_a0_ptr = getelementptr i8, i8 addrspace(1)* %Q, i64 %op_0_a0_byte64
  %op_0_a0_typed = bitcast i8 addrspace(1)* %op_0_a0_ptr to <2 x half> addrspace(1)*
  %op_0_a0_load = load <2 x half>, <2 x half> addrspace(1)* %op_0_a0_typed, align 4
  %op_0_a0_v2 = bitcast <2 x half> %op_0_a0_load to <2 x half>
  %op_0_a0_sram_e0 = extractelement <2 x half> %op_0_a0_v2, i32 0
  %op_0_a0_sram_e1 = extractelement <2 x half> %op_0_a0_v2, i32 1
  %op_0_a0_sram_v0 = insertelement <64 x half> undef, half %op_0_a0_sram_e0, i32 0
  %op_0_a0_sram = insertelement <64 x half> %op_0_a0_sram_v0, half %op_0_a0_sram_e1, i32 1
  %op_0_b0x0_row = add i32 %morton_x, 0
  %op_0_b0x0_col = add i32 %morton_y, 0
  %op_0_b0x0_r_0 = add i32 %op_0_b0x0_row, 0
  %op_0_b0x0_addr_0 = mul i32 %op_0_b0x0_r_0, 16
  %op_0_b0x0_addr2_0 = add i32 %op_0_b0x0_addr_0, %op_0_b0x0_col
  %op_0_b0x0_byte_0 = mul i32 %op_0_b0x0_addr2_0, 2
  %op_0_b0x0_byte64_0 = zext i32 %op_0_b0x0_byte_0 to i64
  %op_0_b0x0_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x0_byte64_0
  %op_0_b0x0_typed_0 = bitcast i8 addrspace(3)* %op_0_b0x0_ptr_0 to half addrspace(3)*
  %op_0_b0x0_load_0 = load half, half addrspace(3)* %op_0_b0x0_typed_0
  %op_0_b0x0_r_1 = add i32 %op_0_b0x0_row, 1
  %op_0_b0x0_addr_1 = mul i32 %op_0_b0x0_r_1, 16
  %op_0_b0x0_addr2_1 = add i32 %op_0_b0x0_addr_1, %op_0_b0x0_col
  %op_0_b0x0_byte_1 = mul i32 %op_0_b0x0_addr2_1, 2
  %op_0_b0x0_byte64_1 = zext i32 %op_0_b0x0_byte_1 to i64
  %op_0_b0x0_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x0_byte64_1
  %op_0_b0x0_typed_1 = bitcast i8 addrspace(3)* %op_0_b0x0_ptr_1 to half addrspace(3)*
  %op_0_b0x0_load_1 = load half, half addrspace(3)* %op_0_b0x0_typed_1
  %op_0_b0x0_v2_a = insertelement <2 x half> undef, half %op_0_b0x0_load_0, i32 0
  %op_0_b0x0_v2 = insertelement <2 x half> %op_0_b0x0_v2_a, half %op_0_b0x0_load_1, i32 1
  %op_0_b0x0_sram_e0 = extractelement <2 x half> %op_0_b0x0_v2, i32 0
  %op_0_b0x0_sram_e1 = extractelement <2 x half> %op_0_b0x0_v2, i32 1
  %op_0_b0x0_sram_v0 = insertelement <64 x half> undef, half %op_0_b0x0_sram_e0, i32 0
  %op_0_b0x0_sram = insertelement <64 x half> %op_0_b0x0_sram_v0, half %op_0_b0x0_sram_e1, i32 1
  %op_0_c0x0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a0_sram, <64 x half> %op_0_b0x0_sram, <64 x float> %s_init_0) #3
  %op_0_b0x1_row = add i32 %morton_x, 8
  %op_0_b0x1_col = add i32 %morton_y, 0
  %op_0_b0x1_r_0 = add i32 %op_0_b0x1_row, 0
  %op_0_b0x1_addr_0 = mul i32 %op_0_b0x1_r_0, 16
  %op_0_b0x1_addr2_0 = add i32 %op_0_b0x1_addr_0, %op_0_b0x1_col
  %op_0_b0x1_byte_0 = mul i32 %op_0_b0x1_addr2_0, 2
  %op_0_b0x1_byte64_0 = zext i32 %op_0_b0x1_byte_0 to i64
  %op_0_b0x1_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x1_byte64_0
  %op_0_b0x1_typed_0 = bitcast i8 addrspace(3)* %op_0_b0x1_ptr_0 to half addrspace(3)*
  %op_0_b0x1_load_0 = load half, half addrspace(3)* %op_0_b0x1_typed_0
  %op_0_b0x1_r_1 = add i32 %op_0_b0x1_row, 1
  %op_0_b0x1_addr_1 = mul i32 %op_0_b0x1_r_1, 16
  %op_0_b0x1_addr2_1 = add i32 %op_0_b0x1_addr_1, %op_0_b0x1_col
  %op_0_b0x1_byte_1 = mul i32 %op_0_b0x1_addr2_1, 2
  %op_0_b0x1_byte64_1 = zext i32 %op_0_b0x1_byte_1 to i64
  %op_0_b0x1_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x1_byte64_1
  %op_0_b0x1_typed_1 = bitcast i8 addrspace(3)* %op_0_b0x1_ptr_1 to half addrspace(3)*
  %op_0_b0x1_load_1 = load half, half addrspace(3)* %op_0_b0x1_typed_1
  %op_0_b0x1_v2_a = insertelement <2 x half> undef, half %op_0_b0x1_load_0, i32 0
  %op_0_b0x1_v2 = insertelement <2 x half> %op_0_b0x1_v2_a, half %op_0_b0x1_load_1, i32 1
  %op_0_b0x1_sram_e0 = extractelement <2 x half> %op_0_b0x1_v2, i32 0
  %op_0_b0x1_sram_e1 = extractelement <2 x half> %op_0_b0x1_v2, i32 1
  %op_0_b0x1_sram_v0 = insertelement <64 x half> undef, half %op_0_b0x1_sram_e0, i32 0
  %op_0_b0x1_sram = insertelement <64 x half> %op_0_b0x1_sram_v0, half %op_0_b0x1_sram_e1, i32 1
  %op_0_c0x1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a0_sram, <64 x half> %op_0_b0x1_sram, <64 x float> %s_init_1) #3
  %op_0_b0x2_row = add i32 %morton_x, 16
  %op_0_b0x2_col = add i32 %morton_y, 0
  %op_0_b0x2_r_0 = add i32 %op_0_b0x2_row, 0
  %op_0_b0x2_addr_0 = mul i32 %op_0_b0x2_r_0, 16
  %op_0_b0x2_addr2_0 = add i32 %op_0_b0x2_addr_0, %op_0_b0x2_col
  %op_0_b0x2_byte_0 = mul i32 %op_0_b0x2_addr2_0, 2
  %op_0_b0x2_byte64_0 = zext i32 %op_0_b0x2_byte_0 to i64
  %op_0_b0x2_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x2_byte64_0
  %op_0_b0x2_typed_0 = bitcast i8 addrspace(3)* %op_0_b0x2_ptr_0 to half addrspace(3)*
  %op_0_b0x2_load_0 = load half, half addrspace(3)* %op_0_b0x2_typed_0
  %op_0_b0x2_r_1 = add i32 %op_0_b0x2_row, 1
  %op_0_b0x2_addr_1 = mul i32 %op_0_b0x2_r_1, 16
  %op_0_b0x2_addr2_1 = add i32 %op_0_b0x2_addr_1, %op_0_b0x2_col
  %op_0_b0x2_byte_1 = mul i32 %op_0_b0x2_addr2_1, 2
  %op_0_b0x2_byte64_1 = zext i32 %op_0_b0x2_byte_1 to i64
  %op_0_b0x2_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x2_byte64_1
  %op_0_b0x2_typed_1 = bitcast i8 addrspace(3)* %op_0_b0x2_ptr_1 to half addrspace(3)*
  %op_0_b0x2_load_1 = load half, half addrspace(3)* %op_0_b0x2_typed_1
  %op_0_b0x2_v2_a = insertelement <2 x half> undef, half %op_0_b0x2_load_0, i32 0
  %op_0_b0x2_v2 = insertelement <2 x half> %op_0_b0x2_v2_a, half %op_0_b0x2_load_1, i32 1
  %op_0_b0x2_sram_e0 = extractelement <2 x half> %op_0_b0x2_v2, i32 0
  %op_0_b0x2_sram_e1 = extractelement <2 x half> %op_0_b0x2_v2, i32 1
  %op_0_b0x2_sram_v0 = insertelement <64 x half> undef, half %op_0_b0x2_sram_e0, i32 0
  %op_0_b0x2_sram = insertelement <64 x half> %op_0_b0x2_sram_v0, half %op_0_b0x2_sram_e1, i32 1
  %op_0_c0x2 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a0_sram, <64 x half> %op_0_b0x2_sram, <64 x float> %s_init_2) #3
  %op_0_b0x3_row = add i32 %morton_x, 24
  %op_0_b0x3_col = add i32 %morton_y, 0
  %op_0_b0x3_r_0 = add i32 %op_0_b0x3_row, 0
  %op_0_b0x3_addr_0 = mul i32 %op_0_b0x3_r_0, 16
  %op_0_b0x3_addr2_0 = add i32 %op_0_b0x3_addr_0, %op_0_b0x3_col
  %op_0_b0x3_byte_0 = mul i32 %op_0_b0x3_addr2_0, 2
  %op_0_b0x3_byte64_0 = zext i32 %op_0_b0x3_byte_0 to i64
  %op_0_b0x3_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x3_byte64_0
  %op_0_b0x3_typed_0 = bitcast i8 addrspace(3)* %op_0_b0x3_ptr_0 to half addrspace(3)*
  %op_0_b0x3_load_0 = load half, half addrspace(3)* %op_0_b0x3_typed_0
  %op_0_b0x3_r_1 = add i32 %op_0_b0x3_row, 1
  %op_0_b0x3_addr_1 = mul i32 %op_0_b0x3_r_1, 16
  %op_0_b0x3_addr2_1 = add i32 %op_0_b0x3_addr_1, %op_0_b0x3_col
  %op_0_b0x3_byte_1 = mul i32 %op_0_b0x3_addr2_1, 2
  %op_0_b0x3_byte64_1 = zext i32 %op_0_b0x3_byte_1 to i64
  %op_0_b0x3_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x3_byte64_1
  %op_0_b0x3_typed_1 = bitcast i8 addrspace(3)* %op_0_b0x3_ptr_1 to half addrspace(3)*
  %op_0_b0x3_load_1 = load half, half addrspace(3)* %op_0_b0x3_typed_1
  %op_0_b0x3_v2_a = insertelement <2 x half> undef, half %op_0_b0x3_load_0, i32 0
  %op_0_b0x3_v2 = insertelement <2 x half> %op_0_b0x3_v2_a, half %op_0_b0x3_load_1, i32 1
  %op_0_b0x3_sram_e0 = extractelement <2 x half> %op_0_b0x3_v2, i32 0
  %op_0_b0x3_sram_e1 = extractelement <2 x half> %op_0_b0x3_v2, i32 1
  %op_0_b0x3_sram_v0 = insertelement <64 x half> undef, half %op_0_b0x3_sram_e0, i32 0
  %op_0_b0x3_sram = insertelement <64 x half> %op_0_b0x3_sram_v0, half %op_0_b0x3_sram_e1, i32 1
  %op_0_c0x3 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a0_sram, <64 x half> %op_0_b0x3_sram, <64 x float> %s_init_3) #3
  %op_0_b0x4_row = add i32 %morton_x, 32
  %op_0_b0x4_col = add i32 %morton_y, 0
  %op_0_b0x4_r_0 = add i32 %op_0_b0x4_row, 0
  %op_0_b0x4_addr_0 = mul i32 %op_0_b0x4_r_0, 16
  %op_0_b0x4_addr2_0 = add i32 %op_0_b0x4_addr_0, %op_0_b0x4_col
  %op_0_b0x4_byte_0 = mul i32 %op_0_b0x4_addr2_0, 2
  %op_0_b0x4_byte64_0 = zext i32 %op_0_b0x4_byte_0 to i64
  %op_0_b0x4_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x4_byte64_0
  %op_0_b0x4_typed_0 = bitcast i8 addrspace(3)* %op_0_b0x4_ptr_0 to half addrspace(3)*
  %op_0_b0x4_load_0 = load half, half addrspace(3)* %op_0_b0x4_typed_0
  %op_0_b0x4_r_1 = add i32 %op_0_b0x4_row, 1
  %op_0_b0x4_addr_1 = mul i32 %op_0_b0x4_r_1, 16
  %op_0_b0x4_addr2_1 = add i32 %op_0_b0x4_addr_1, %op_0_b0x4_col
  %op_0_b0x4_byte_1 = mul i32 %op_0_b0x4_addr2_1, 2
  %op_0_b0x4_byte64_1 = zext i32 %op_0_b0x4_byte_1 to i64
  %op_0_b0x4_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x4_byte64_1
  %op_0_b0x4_typed_1 = bitcast i8 addrspace(3)* %op_0_b0x4_ptr_1 to half addrspace(3)*
  %op_0_b0x4_load_1 = load half, half addrspace(3)* %op_0_b0x4_typed_1
  %op_0_b0x4_v2_a = insertelement <2 x half> undef, half %op_0_b0x4_load_0, i32 0
  %op_0_b0x4_v2 = insertelement <2 x half> %op_0_b0x4_v2_a, half %op_0_b0x4_load_1, i32 1
  %op_0_b0x4_sram_e0 = extractelement <2 x half> %op_0_b0x4_v2, i32 0
  %op_0_b0x4_sram_e1 = extractelement <2 x half> %op_0_b0x4_v2, i32 1
  %op_0_b0x4_sram_v0 = insertelement <64 x half> undef, half %op_0_b0x4_sram_e0, i32 0
  %op_0_b0x4_sram = insertelement <64 x half> %op_0_b0x4_sram_v0, half %op_0_b0x4_sram_e1, i32 1
  %op_0_c0x4 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a0_sram, <64 x half> %op_0_b0x4_sram, <64 x float> %s_init_4) #3
  %op_0_b0x5_row = add i32 %morton_x, 40
  %op_0_b0x5_col = add i32 %morton_y, 0
  %op_0_b0x5_r_0 = add i32 %op_0_b0x5_row, 0
  %op_0_b0x5_addr_0 = mul i32 %op_0_b0x5_r_0, 16
  %op_0_b0x5_addr2_0 = add i32 %op_0_b0x5_addr_0, %op_0_b0x5_col
  %op_0_b0x5_byte_0 = mul i32 %op_0_b0x5_addr2_0, 2
  %op_0_b0x5_byte64_0 = zext i32 %op_0_b0x5_byte_0 to i64
  %op_0_b0x5_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x5_byte64_0
  %op_0_b0x5_typed_0 = bitcast i8 addrspace(3)* %op_0_b0x5_ptr_0 to half addrspace(3)*
  %op_0_b0x5_load_0 = load half, half addrspace(3)* %op_0_b0x5_typed_0
  %op_0_b0x5_r_1 = add i32 %op_0_b0x5_row, 1
  %op_0_b0x5_addr_1 = mul i32 %op_0_b0x5_r_1, 16
  %op_0_b0x5_addr2_1 = add i32 %op_0_b0x5_addr_1, %op_0_b0x5_col
  %op_0_b0x5_byte_1 = mul i32 %op_0_b0x5_addr2_1, 2
  %op_0_b0x5_byte64_1 = zext i32 %op_0_b0x5_byte_1 to i64
  %op_0_b0x5_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x5_byte64_1
  %op_0_b0x5_typed_1 = bitcast i8 addrspace(3)* %op_0_b0x5_ptr_1 to half addrspace(3)*
  %op_0_b0x5_load_1 = load half, half addrspace(3)* %op_0_b0x5_typed_1
  %op_0_b0x5_v2_a = insertelement <2 x half> undef, half %op_0_b0x5_load_0, i32 0
  %op_0_b0x5_v2 = insertelement <2 x half> %op_0_b0x5_v2_a, half %op_0_b0x5_load_1, i32 1
  %op_0_b0x5_sram_e0 = extractelement <2 x half> %op_0_b0x5_v2, i32 0
  %op_0_b0x5_sram_e1 = extractelement <2 x half> %op_0_b0x5_v2, i32 1
  %op_0_b0x5_sram_v0 = insertelement <64 x half> undef, half %op_0_b0x5_sram_e0, i32 0
  %op_0_b0x5_sram = insertelement <64 x half> %op_0_b0x5_sram_v0, half %op_0_b0x5_sram_e1, i32 1
  %op_0_c0x5 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a0_sram, <64 x half> %op_0_b0x5_sram, <64 x float> %s_init_5) #3
  %op_0_b0x6_row = add i32 %morton_x, 48
  %op_0_b0x6_col = add i32 %morton_y, 0
  %op_0_b0x6_r_0 = add i32 %op_0_b0x6_row, 0
  %op_0_b0x6_addr_0 = mul i32 %op_0_b0x6_r_0, 16
  %op_0_b0x6_addr2_0 = add i32 %op_0_b0x6_addr_0, %op_0_b0x6_col
  %op_0_b0x6_byte_0 = mul i32 %op_0_b0x6_addr2_0, 2
  %op_0_b0x6_byte64_0 = zext i32 %op_0_b0x6_byte_0 to i64
  %op_0_b0x6_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x6_byte64_0
  %op_0_b0x6_typed_0 = bitcast i8 addrspace(3)* %op_0_b0x6_ptr_0 to half addrspace(3)*
  %op_0_b0x6_load_0 = load half, half addrspace(3)* %op_0_b0x6_typed_0
  %op_0_b0x6_r_1 = add i32 %op_0_b0x6_row, 1
  %op_0_b0x6_addr_1 = mul i32 %op_0_b0x6_r_1, 16
  %op_0_b0x6_addr2_1 = add i32 %op_0_b0x6_addr_1, %op_0_b0x6_col
  %op_0_b0x6_byte_1 = mul i32 %op_0_b0x6_addr2_1, 2
  %op_0_b0x6_byte64_1 = zext i32 %op_0_b0x6_byte_1 to i64
  %op_0_b0x6_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x6_byte64_1
  %op_0_b0x6_typed_1 = bitcast i8 addrspace(3)* %op_0_b0x6_ptr_1 to half addrspace(3)*
  %op_0_b0x6_load_1 = load half, half addrspace(3)* %op_0_b0x6_typed_1
  %op_0_b0x6_v2_a = insertelement <2 x half> undef, half %op_0_b0x6_load_0, i32 0
  %op_0_b0x6_v2 = insertelement <2 x half> %op_0_b0x6_v2_a, half %op_0_b0x6_load_1, i32 1
  %op_0_b0x6_sram_e0 = extractelement <2 x half> %op_0_b0x6_v2, i32 0
  %op_0_b0x6_sram_e1 = extractelement <2 x half> %op_0_b0x6_v2, i32 1
  %op_0_b0x6_sram_v0 = insertelement <64 x half> undef, half %op_0_b0x6_sram_e0, i32 0
  %op_0_b0x6_sram = insertelement <64 x half> %op_0_b0x6_sram_v0, half %op_0_b0x6_sram_e1, i32 1
  %op_0_c0x6 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a0_sram, <64 x half> %op_0_b0x6_sram, <64 x float> %s_init_6) #3
  %op_0_b0x7_row = add i32 %morton_x, 56
  %op_0_b0x7_col = add i32 %morton_y, 0
  %op_0_b0x7_r_0 = add i32 %op_0_b0x7_row, 0
  %op_0_b0x7_addr_0 = mul i32 %op_0_b0x7_r_0, 16
  %op_0_b0x7_addr2_0 = add i32 %op_0_b0x7_addr_0, %op_0_b0x7_col
  %op_0_b0x7_byte_0 = mul i32 %op_0_b0x7_addr2_0, 2
  %op_0_b0x7_byte64_0 = zext i32 %op_0_b0x7_byte_0 to i64
  %op_0_b0x7_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x7_byte64_0
  %op_0_b0x7_typed_0 = bitcast i8 addrspace(3)* %op_0_b0x7_ptr_0 to half addrspace(3)*
  %op_0_b0x7_load_0 = load half, half addrspace(3)* %op_0_b0x7_typed_0
  %op_0_b0x7_r_1 = add i32 %op_0_b0x7_row, 1
  %op_0_b0x7_addr_1 = mul i32 %op_0_b0x7_r_1, 16
  %op_0_b0x7_addr2_1 = add i32 %op_0_b0x7_addr_1, %op_0_b0x7_col
  %op_0_b0x7_byte_1 = mul i32 %op_0_b0x7_addr2_1, 2
  %op_0_b0x7_byte64_1 = zext i32 %op_0_b0x7_byte_1 to i64
  %op_0_b0x7_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b0x7_byte64_1
  %op_0_b0x7_typed_1 = bitcast i8 addrspace(3)* %op_0_b0x7_ptr_1 to half addrspace(3)*
  %op_0_b0x7_load_1 = load half, half addrspace(3)* %op_0_b0x7_typed_1
  %op_0_b0x7_v2_a = insertelement <2 x half> undef, half %op_0_b0x7_load_0, i32 0
  %op_0_b0x7_v2 = insertelement <2 x half> %op_0_b0x7_v2_a, half %op_0_b0x7_load_1, i32 1
  %op_0_b0x7_sram_e0 = extractelement <2 x half> %op_0_b0x7_v2, i32 0
  %op_0_b0x7_sram_e1 = extractelement <2 x half> %op_0_b0x7_v2, i32 1
  %op_0_b0x7_sram_v0 = insertelement <64 x half> undef, half %op_0_b0x7_sram_e0, i32 0
  %op_0_b0x7_sram = insertelement <64 x half> %op_0_b0x7_sram_v0, half %op_0_b0x7_sram_e1, i32 1
  %op_0_c0x7 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a0_sram, <64 x half> %op_0_b0x7_sram, <64 x float> %s_init_7) #3
  %op_0_a1_seq = add i32 %clamped_par_off, 0
  %op_0_a1_head = add i32 %morton_x, 8
  %op_0_a1_addr = mul i32 %op_0_a1_seq, 64
  %op_0_a1_addr2 = add i32 %op_0_a1_addr, %op_0_a1_head
  %op_0_a1_byte = mul i32 %op_0_a1_addr2, 2
  %op_0_a1_byte64 = zext i32 %op_0_a1_byte to i64
  %op_0_a1_ptr = getelementptr i8, i8 addrspace(1)* %Q, i64 %op_0_a1_byte64
  %op_0_a1_typed = bitcast i8 addrspace(1)* %op_0_a1_ptr to <2 x half> addrspace(1)*
  %op_0_a1_load = load <2 x half>, <2 x half> addrspace(1)* %op_0_a1_typed, align 4
  %op_0_a1_v2 = bitcast <2 x half> %op_0_a1_load to <2 x half>
  %op_0_a1_sram_e0 = extractelement <2 x half> %op_0_a1_v2, i32 0
  %op_0_a1_sram_e1 = extractelement <2 x half> %op_0_a1_v2, i32 1
  %op_0_a1_sram_v0 = insertelement <64 x half> undef, half %op_0_a1_sram_e0, i32 0
  %op_0_a1_sram = insertelement <64 x half> %op_0_a1_sram_v0, half %op_0_a1_sram_e1, i32 1
  %op_0_b1x0_row = add i32 %morton_x, 0
  %op_0_b1x0_col = add i32 %morton_y, 8
  %op_0_b1x0_r_0 = add i32 %op_0_b1x0_row, 0
  %op_0_b1x0_addr_0 = mul i32 %op_0_b1x0_r_0, 16
  %op_0_b1x0_addr2_0 = add i32 %op_0_b1x0_addr_0, %op_0_b1x0_col
  %op_0_b1x0_byte_0 = mul i32 %op_0_b1x0_addr2_0, 2
  %op_0_b1x0_byte64_0 = zext i32 %op_0_b1x0_byte_0 to i64
  %op_0_b1x0_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x0_byte64_0
  %op_0_b1x0_typed_0 = bitcast i8 addrspace(3)* %op_0_b1x0_ptr_0 to half addrspace(3)*
  %op_0_b1x0_load_0 = load half, half addrspace(3)* %op_0_b1x0_typed_0
  %op_0_b1x0_r_1 = add i32 %op_0_b1x0_row, 1
  %op_0_b1x0_addr_1 = mul i32 %op_0_b1x0_r_1, 16
  %op_0_b1x0_addr2_1 = add i32 %op_0_b1x0_addr_1, %op_0_b1x0_col
  %op_0_b1x0_byte_1 = mul i32 %op_0_b1x0_addr2_1, 2
  %op_0_b1x0_byte64_1 = zext i32 %op_0_b1x0_byte_1 to i64
  %op_0_b1x0_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x0_byte64_1
  %op_0_b1x0_typed_1 = bitcast i8 addrspace(3)* %op_0_b1x0_ptr_1 to half addrspace(3)*
  %op_0_b1x0_load_1 = load half, half addrspace(3)* %op_0_b1x0_typed_1
  %op_0_b1x0_v2_a = insertelement <2 x half> undef, half %op_0_b1x0_load_0, i32 0
  %op_0_b1x0_v2 = insertelement <2 x half> %op_0_b1x0_v2_a, half %op_0_b1x0_load_1, i32 1
  %op_0_b1x0_sram_e0 = extractelement <2 x half> %op_0_b1x0_v2, i32 0
  %op_0_b1x0_sram_e1 = extractelement <2 x half> %op_0_b1x0_v2, i32 1
  %op_0_b1x0_sram_v0 = insertelement <64 x half> undef, half %op_0_b1x0_sram_e0, i32 0
  %op_0_b1x0_sram = insertelement <64 x half> %op_0_b1x0_sram_v0, half %op_0_b1x0_sram_e1, i32 1
  %op_0_c1x0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a1_sram, <64 x half> %op_0_b1x0_sram, <64 x float> %op_0_c0x0) #3
  %op_0_b1x1_row = add i32 %morton_x, 8
  %op_0_b1x1_col = add i32 %morton_y, 8
  %op_0_b1x1_r_0 = add i32 %op_0_b1x1_row, 0
  %op_0_b1x1_addr_0 = mul i32 %op_0_b1x1_r_0, 16
  %op_0_b1x1_addr2_0 = add i32 %op_0_b1x1_addr_0, %op_0_b1x1_col
  %op_0_b1x1_byte_0 = mul i32 %op_0_b1x1_addr2_0, 2
  %op_0_b1x1_byte64_0 = zext i32 %op_0_b1x1_byte_0 to i64
  %op_0_b1x1_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x1_byte64_0
  %op_0_b1x1_typed_0 = bitcast i8 addrspace(3)* %op_0_b1x1_ptr_0 to half addrspace(3)*
  %op_0_b1x1_load_0 = load half, half addrspace(3)* %op_0_b1x1_typed_0
  %op_0_b1x1_r_1 = add i32 %op_0_b1x1_row, 1
  %op_0_b1x1_addr_1 = mul i32 %op_0_b1x1_r_1, 16
  %op_0_b1x1_addr2_1 = add i32 %op_0_b1x1_addr_1, %op_0_b1x1_col
  %op_0_b1x1_byte_1 = mul i32 %op_0_b1x1_addr2_1, 2
  %op_0_b1x1_byte64_1 = zext i32 %op_0_b1x1_byte_1 to i64
  %op_0_b1x1_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x1_byte64_1
  %op_0_b1x1_typed_1 = bitcast i8 addrspace(3)* %op_0_b1x1_ptr_1 to half addrspace(3)*
  %op_0_b1x1_load_1 = load half, half addrspace(3)* %op_0_b1x1_typed_1
  %op_0_b1x1_v2_a = insertelement <2 x half> undef, half %op_0_b1x1_load_0, i32 0
  %op_0_b1x1_v2 = insertelement <2 x half> %op_0_b1x1_v2_a, half %op_0_b1x1_load_1, i32 1
  %op_0_b1x1_sram_e0 = extractelement <2 x half> %op_0_b1x1_v2, i32 0
  %op_0_b1x1_sram_e1 = extractelement <2 x half> %op_0_b1x1_v2, i32 1
  %op_0_b1x1_sram_v0 = insertelement <64 x half> undef, half %op_0_b1x1_sram_e0, i32 0
  %op_0_b1x1_sram = insertelement <64 x half> %op_0_b1x1_sram_v0, half %op_0_b1x1_sram_e1, i32 1
  %op_0_c1x1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a1_sram, <64 x half> %op_0_b1x1_sram, <64 x float> %op_0_c0x1) #3
  %op_0_b1x2_row = add i32 %morton_x, 16
  %op_0_b1x2_col = add i32 %morton_y, 8
  %op_0_b1x2_r_0 = add i32 %op_0_b1x2_row, 0
  %op_0_b1x2_addr_0 = mul i32 %op_0_b1x2_r_0, 16
  %op_0_b1x2_addr2_0 = add i32 %op_0_b1x2_addr_0, %op_0_b1x2_col
  %op_0_b1x2_byte_0 = mul i32 %op_0_b1x2_addr2_0, 2
  %op_0_b1x2_byte64_0 = zext i32 %op_0_b1x2_byte_0 to i64
  %op_0_b1x2_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x2_byte64_0
  %op_0_b1x2_typed_0 = bitcast i8 addrspace(3)* %op_0_b1x2_ptr_0 to half addrspace(3)*
  %op_0_b1x2_load_0 = load half, half addrspace(3)* %op_0_b1x2_typed_0
  %op_0_b1x2_r_1 = add i32 %op_0_b1x2_row, 1
  %op_0_b1x2_addr_1 = mul i32 %op_0_b1x2_r_1, 16
  %op_0_b1x2_addr2_1 = add i32 %op_0_b1x2_addr_1, %op_0_b1x2_col
  %op_0_b1x2_byte_1 = mul i32 %op_0_b1x2_addr2_1, 2
  %op_0_b1x2_byte64_1 = zext i32 %op_0_b1x2_byte_1 to i64
  %op_0_b1x2_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x2_byte64_1
  %op_0_b1x2_typed_1 = bitcast i8 addrspace(3)* %op_0_b1x2_ptr_1 to half addrspace(3)*
  %op_0_b1x2_load_1 = load half, half addrspace(3)* %op_0_b1x2_typed_1
  %op_0_b1x2_v2_a = insertelement <2 x half> undef, half %op_0_b1x2_load_0, i32 0
  %op_0_b1x2_v2 = insertelement <2 x half> %op_0_b1x2_v2_a, half %op_0_b1x2_load_1, i32 1
  %op_0_b1x2_sram_e0 = extractelement <2 x half> %op_0_b1x2_v2, i32 0
  %op_0_b1x2_sram_e1 = extractelement <2 x half> %op_0_b1x2_v2, i32 1
  %op_0_b1x2_sram_v0 = insertelement <64 x half> undef, half %op_0_b1x2_sram_e0, i32 0
  %op_0_b1x2_sram = insertelement <64 x half> %op_0_b1x2_sram_v0, half %op_0_b1x2_sram_e1, i32 1
  %op_0_c1x2 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a1_sram, <64 x half> %op_0_b1x2_sram, <64 x float> %op_0_c0x2) #3
  %op_0_b1x3_row = add i32 %morton_x, 24
  %op_0_b1x3_col = add i32 %morton_y, 8
  %op_0_b1x3_r_0 = add i32 %op_0_b1x3_row, 0
  %op_0_b1x3_addr_0 = mul i32 %op_0_b1x3_r_0, 16
  %op_0_b1x3_addr2_0 = add i32 %op_0_b1x3_addr_0, %op_0_b1x3_col
  %op_0_b1x3_byte_0 = mul i32 %op_0_b1x3_addr2_0, 2
  %op_0_b1x3_byte64_0 = zext i32 %op_0_b1x3_byte_0 to i64
  %op_0_b1x3_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x3_byte64_0
  %op_0_b1x3_typed_0 = bitcast i8 addrspace(3)* %op_0_b1x3_ptr_0 to half addrspace(3)*
  %op_0_b1x3_load_0 = load half, half addrspace(3)* %op_0_b1x3_typed_0
  %op_0_b1x3_r_1 = add i32 %op_0_b1x3_row, 1
  %op_0_b1x3_addr_1 = mul i32 %op_0_b1x3_r_1, 16
  %op_0_b1x3_addr2_1 = add i32 %op_0_b1x3_addr_1, %op_0_b1x3_col
  %op_0_b1x3_byte_1 = mul i32 %op_0_b1x3_addr2_1, 2
  %op_0_b1x3_byte64_1 = zext i32 %op_0_b1x3_byte_1 to i64
  %op_0_b1x3_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x3_byte64_1
  %op_0_b1x3_typed_1 = bitcast i8 addrspace(3)* %op_0_b1x3_ptr_1 to half addrspace(3)*
  %op_0_b1x3_load_1 = load half, half addrspace(3)* %op_0_b1x3_typed_1
  %op_0_b1x3_v2_a = insertelement <2 x half> undef, half %op_0_b1x3_load_0, i32 0
  %op_0_b1x3_v2 = insertelement <2 x half> %op_0_b1x3_v2_a, half %op_0_b1x3_load_1, i32 1
  %op_0_b1x3_sram_e0 = extractelement <2 x half> %op_0_b1x3_v2, i32 0
  %op_0_b1x3_sram_e1 = extractelement <2 x half> %op_0_b1x3_v2, i32 1
  %op_0_b1x3_sram_v0 = insertelement <64 x half> undef, half %op_0_b1x3_sram_e0, i32 0
  %op_0_b1x3_sram = insertelement <64 x half> %op_0_b1x3_sram_v0, half %op_0_b1x3_sram_e1, i32 1
  %op_0_c1x3 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a1_sram, <64 x half> %op_0_b1x3_sram, <64 x float> %op_0_c0x3) #3
  %op_0_b1x4_row = add i32 %morton_x, 32
  %op_0_b1x4_col = add i32 %morton_y, 8
  %op_0_b1x4_r_0 = add i32 %op_0_b1x4_row, 0
  %op_0_b1x4_addr_0 = mul i32 %op_0_b1x4_r_0, 16
  %op_0_b1x4_addr2_0 = add i32 %op_0_b1x4_addr_0, %op_0_b1x4_col
  %op_0_b1x4_byte_0 = mul i32 %op_0_b1x4_addr2_0, 2
  %op_0_b1x4_byte64_0 = zext i32 %op_0_b1x4_byte_0 to i64
  %op_0_b1x4_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x4_byte64_0
  %op_0_b1x4_typed_0 = bitcast i8 addrspace(3)* %op_0_b1x4_ptr_0 to half addrspace(3)*
  %op_0_b1x4_load_0 = load half, half addrspace(3)* %op_0_b1x4_typed_0
  %op_0_b1x4_r_1 = add i32 %op_0_b1x4_row, 1
  %op_0_b1x4_addr_1 = mul i32 %op_0_b1x4_r_1, 16
  %op_0_b1x4_addr2_1 = add i32 %op_0_b1x4_addr_1, %op_0_b1x4_col
  %op_0_b1x4_byte_1 = mul i32 %op_0_b1x4_addr2_1, 2
  %op_0_b1x4_byte64_1 = zext i32 %op_0_b1x4_byte_1 to i64
  %op_0_b1x4_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x4_byte64_1
  %op_0_b1x4_typed_1 = bitcast i8 addrspace(3)* %op_0_b1x4_ptr_1 to half addrspace(3)*
  %op_0_b1x4_load_1 = load half, half addrspace(3)* %op_0_b1x4_typed_1
  %op_0_b1x4_v2_a = insertelement <2 x half> undef, half %op_0_b1x4_load_0, i32 0
  %op_0_b1x4_v2 = insertelement <2 x half> %op_0_b1x4_v2_a, half %op_0_b1x4_load_1, i32 1
  %op_0_b1x4_sram_e0 = extractelement <2 x half> %op_0_b1x4_v2, i32 0
  %op_0_b1x4_sram_e1 = extractelement <2 x half> %op_0_b1x4_v2, i32 1
  %op_0_b1x4_sram_v0 = insertelement <64 x half> undef, half %op_0_b1x4_sram_e0, i32 0
  %op_0_b1x4_sram = insertelement <64 x half> %op_0_b1x4_sram_v0, half %op_0_b1x4_sram_e1, i32 1
  %op_0_c1x4 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a1_sram, <64 x half> %op_0_b1x4_sram, <64 x float> %op_0_c0x4) #3
  %op_0_b1x5_row = add i32 %morton_x, 40
  %op_0_b1x5_col = add i32 %morton_y, 8
  %op_0_b1x5_r_0 = add i32 %op_0_b1x5_row, 0
  %op_0_b1x5_addr_0 = mul i32 %op_0_b1x5_r_0, 16
  %op_0_b1x5_addr2_0 = add i32 %op_0_b1x5_addr_0, %op_0_b1x5_col
  %op_0_b1x5_byte_0 = mul i32 %op_0_b1x5_addr2_0, 2
  %op_0_b1x5_byte64_0 = zext i32 %op_0_b1x5_byte_0 to i64
  %op_0_b1x5_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x5_byte64_0
  %op_0_b1x5_typed_0 = bitcast i8 addrspace(3)* %op_0_b1x5_ptr_0 to half addrspace(3)*
  %op_0_b1x5_load_0 = load half, half addrspace(3)* %op_0_b1x5_typed_0
  %op_0_b1x5_r_1 = add i32 %op_0_b1x5_row, 1
  %op_0_b1x5_addr_1 = mul i32 %op_0_b1x5_r_1, 16
  %op_0_b1x5_addr2_1 = add i32 %op_0_b1x5_addr_1, %op_0_b1x5_col
  %op_0_b1x5_byte_1 = mul i32 %op_0_b1x5_addr2_1, 2
  %op_0_b1x5_byte64_1 = zext i32 %op_0_b1x5_byte_1 to i64
  %op_0_b1x5_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x5_byte64_1
  %op_0_b1x5_typed_1 = bitcast i8 addrspace(3)* %op_0_b1x5_ptr_1 to half addrspace(3)*
  %op_0_b1x5_load_1 = load half, half addrspace(3)* %op_0_b1x5_typed_1
  %op_0_b1x5_v2_a = insertelement <2 x half> undef, half %op_0_b1x5_load_0, i32 0
  %op_0_b1x5_v2 = insertelement <2 x half> %op_0_b1x5_v2_a, half %op_0_b1x5_load_1, i32 1
  %op_0_b1x5_sram_e0 = extractelement <2 x half> %op_0_b1x5_v2, i32 0
  %op_0_b1x5_sram_e1 = extractelement <2 x half> %op_0_b1x5_v2, i32 1
  %op_0_b1x5_sram_v0 = insertelement <64 x half> undef, half %op_0_b1x5_sram_e0, i32 0
  %op_0_b1x5_sram = insertelement <64 x half> %op_0_b1x5_sram_v0, half %op_0_b1x5_sram_e1, i32 1
  %op_0_c1x5 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a1_sram, <64 x half> %op_0_b1x5_sram, <64 x float> %op_0_c0x5) #3
  %op_0_b1x6_row = add i32 %morton_x, 48
  %op_0_b1x6_col = add i32 %morton_y, 8
  %op_0_b1x6_r_0 = add i32 %op_0_b1x6_row, 0
  %op_0_b1x6_addr_0 = mul i32 %op_0_b1x6_r_0, 16
  %op_0_b1x6_addr2_0 = add i32 %op_0_b1x6_addr_0, %op_0_b1x6_col
  %op_0_b1x6_byte_0 = mul i32 %op_0_b1x6_addr2_0, 2
  %op_0_b1x6_byte64_0 = zext i32 %op_0_b1x6_byte_0 to i64
  %op_0_b1x6_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x6_byte64_0
  %op_0_b1x6_typed_0 = bitcast i8 addrspace(3)* %op_0_b1x6_ptr_0 to half addrspace(3)*
  %op_0_b1x6_load_0 = load half, half addrspace(3)* %op_0_b1x6_typed_0
  %op_0_b1x6_r_1 = add i32 %op_0_b1x6_row, 1
  %op_0_b1x6_addr_1 = mul i32 %op_0_b1x6_r_1, 16
  %op_0_b1x6_addr2_1 = add i32 %op_0_b1x6_addr_1, %op_0_b1x6_col
  %op_0_b1x6_byte_1 = mul i32 %op_0_b1x6_addr2_1, 2
  %op_0_b1x6_byte64_1 = zext i32 %op_0_b1x6_byte_1 to i64
  %op_0_b1x6_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x6_byte64_1
  %op_0_b1x6_typed_1 = bitcast i8 addrspace(3)* %op_0_b1x6_ptr_1 to half addrspace(3)*
  %op_0_b1x6_load_1 = load half, half addrspace(3)* %op_0_b1x6_typed_1
  %op_0_b1x6_v2_a = insertelement <2 x half> undef, half %op_0_b1x6_load_0, i32 0
  %op_0_b1x6_v2 = insertelement <2 x half> %op_0_b1x6_v2_a, half %op_0_b1x6_load_1, i32 1
  %op_0_b1x6_sram_e0 = extractelement <2 x half> %op_0_b1x6_v2, i32 0
  %op_0_b1x6_sram_e1 = extractelement <2 x half> %op_0_b1x6_v2, i32 1
  %op_0_b1x6_sram_v0 = insertelement <64 x half> undef, half %op_0_b1x6_sram_e0, i32 0
  %op_0_b1x6_sram = insertelement <64 x half> %op_0_b1x6_sram_v0, half %op_0_b1x6_sram_e1, i32 1
  %op_0_c1x6 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a1_sram, <64 x half> %op_0_b1x6_sram, <64 x float> %op_0_c0x6) #3
  %op_0_b1x7_row = add i32 %morton_x, 56
  %op_0_b1x7_col = add i32 %morton_y, 8
  %op_0_b1x7_r_0 = add i32 %op_0_b1x7_row, 0
  %op_0_b1x7_addr_0 = mul i32 %op_0_b1x7_r_0, 16
  %op_0_b1x7_addr2_0 = add i32 %op_0_b1x7_addr_0, %op_0_b1x7_col
  %op_0_b1x7_byte_0 = mul i32 %op_0_b1x7_addr2_0, 2
  %op_0_b1x7_byte64_0 = zext i32 %op_0_b1x7_byte_0 to i64
  %op_0_b1x7_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x7_byte64_0
  %op_0_b1x7_typed_0 = bitcast i8 addrspace(3)* %op_0_b1x7_ptr_0 to half addrspace(3)*
  %op_0_b1x7_load_0 = load half, half addrspace(3)* %op_0_b1x7_typed_0
  %op_0_b1x7_r_1 = add i32 %op_0_b1x7_row, 1
  %op_0_b1x7_addr_1 = mul i32 %op_0_b1x7_r_1, 16
  %op_0_b1x7_addr2_1 = add i32 %op_0_b1x7_addr_1, %op_0_b1x7_col
  %op_0_b1x7_byte_1 = mul i32 %op_0_b1x7_addr2_1, 2
  %op_0_b1x7_byte64_1 = zext i32 %op_0_b1x7_byte_1 to i64
  %op_0_b1x7_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_0_b1x7_byte64_1
  %op_0_b1x7_typed_1 = bitcast i8 addrspace(3)* %op_0_b1x7_ptr_1 to half addrspace(3)*
  %op_0_b1x7_load_1 = load half, half addrspace(3)* %op_0_b1x7_typed_1
  %op_0_b1x7_v2_a = insertelement <2 x half> undef, half %op_0_b1x7_load_0, i32 0
  %op_0_b1x7_v2 = insertelement <2 x half> %op_0_b1x7_v2_a, half %op_0_b1x7_load_1, i32 1
  %op_0_b1x7_sram_e0 = extractelement <2 x half> %op_0_b1x7_v2, i32 0
  %op_0_b1x7_sram_e1 = extractelement <2 x half> %op_0_b1x7_v2, i32 1
  %op_0_b1x7_sram_v0 = insertelement <64 x half> undef, half %op_0_b1x7_sram_e0, i32 0
  %op_0_b1x7_sram = insertelement <64 x half> %op_0_b1x7_sram_v0, half %op_0_b1x7_sram_e1, i32 1
  %op_0_c1x7 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_0_a1_sram, <64 x half> %op_0_b1x7_sram, <64 x float> %op_0_c0x7) #3
  call void @air.wg.barrier(i32 2, i32 1)

  ; === Sync copy (op_1_b) — all threads cooperative ===
  %op_1_bsc_t0 = mul i32 %sidx, 32
  %op_1_bsc_tid = add i32 %op_1_bsc_t0, %lane_id
  %op_1_bsc_drem = sub i32 64, 16
  %op_1_bsc_dcmp = icmp ult i32 %op_1_bsc_drem, 16
  %op_1_bsc_dsrc = select i1 %op_1_bsc_dcmp, i32 %op_1_bsc_drem, i32 16
  %op_1_bsc_srem = sub i32 64, %c
  %op_1_bsc_scmp = icmp ult i32 %op_1_bsc_srem, 64
  %op_1_bsc_ssrc = select i1 %op_1_bsc_scmp, i32 %op_1_bsc_srem, i32 64
  br label %op_1_bsc_pre

op_1_bsc_pre:
  br label %op_1_bsc_hdr

op_1_bsc_hdr:
  %op_1_bsc_i = phi i32 [%op_1_bsc_tid, %op_1_bsc_pre], [%op_1_bsc_inx, %op_1_bsc_st]
  %op_1_bsc_done = icmp uge i32 %op_1_bsc_i, 1024
  br i1 %op_1_bsc_done, label %op_1_bsc_end, label %op_1_bsc_body

op_1_bsc_body:
  %op_1_bsc_row = lshr i32 %op_1_bsc_i, 4
  %op_1_bsc_col = and i32 %op_1_bsc_i, 15
  %op_1_bsc_rok = icmp ult i32 %op_1_bsc_row, %op_1_bsc_ssrc
  %op_1_bsc_cok = icmp ult i32 %op_1_bsc_col, %op_1_bsc_dsrc
  %op_1_bsc_ib = and i1 %op_1_bsc_rok, %op_1_bsc_cok
  br i1 %op_1_bsc_ib, label %op_1_bsc_ld, label %op_1_bsc_zr

op_1_bsc_ld:
  %op_1_bsc_sr = add i32 %c, %op_1_bsc_row
  %op_1_bsc_sa = mul i32 %op_1_bsc_sr, 64
  %op_1_bsc_sc = add i32 16, %op_1_bsc_col
  %op_1_bsc_sad = add i32 %op_1_bsc_sa, %op_1_bsc_sc
  %op_1_bsc_soff = zext i32 %op_1_bsc_sad to i64
  %op_1_bsc_sbyt = mul i64 %op_1_bsc_soff, 2
  %op_1_bsc_sp = getelementptr i8, i8 addrspace(1)* %K, i64 %op_1_bsc_sbyt
  %op_1_bsc_spt = bitcast i8 addrspace(1)* %op_1_bsc_sp to i16 addrspace(1)*
  %op_1_bsc_lv = load i16, i16 addrspace(1)* %op_1_bsc_spt
  br label %op_1_bsc_st

op_1_bsc_zr:
  br label %op_1_bsc_st

op_1_bsc_st:
  %op_1_bsc_val = phi i16 [%op_1_bsc_lv, %op_1_bsc_ld], [0, %op_1_bsc_zr]
  %op_1_bsc_tr = mul i32 %op_1_bsc_row, 16
  %op_1_bsc_ta = add i32 %op_1_bsc_tr, %op_1_bsc_col
  %op_1_bsc_tb = mul i32 %op_1_bsc_ta, 2
  %op_1_bsc_tb64 = zext i32 %op_1_bsc_tb to i64
  %op_1_bsc_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_bsc_tb64
  %op_1_bsc_tpt = bitcast i8 addrspace(3)* %op_1_bsc_tp to i16 addrspace(3)*
  store i16 %op_1_bsc_val, i16 addrspace(3)* %op_1_bsc_tpt
  %op_1_bsc_inx = add i32 %op_1_bsc_i, 64
  br label %op_1_bsc_hdr

op_1_bsc_end:
  call void @air.wg.barrier(i32 2, i32 1)

  %op_1_a0_seq = add i32 %clamped_par_off, 0
  %op_1_a0_head = add i32 %morton_x, 16
  %op_1_a0_addr = mul i32 %op_1_a0_seq, 64
  %op_1_a0_addr2 = add i32 %op_1_a0_addr, %op_1_a0_head
  %op_1_a0_byte = mul i32 %op_1_a0_addr2, 2
  %op_1_a0_byte64 = zext i32 %op_1_a0_byte to i64
  %op_1_a0_ptr = getelementptr i8, i8 addrspace(1)* %Q, i64 %op_1_a0_byte64
  %op_1_a0_typed = bitcast i8 addrspace(1)* %op_1_a0_ptr to <2 x half> addrspace(1)*
  %op_1_a0_load = load <2 x half>, <2 x half> addrspace(1)* %op_1_a0_typed, align 4
  %op_1_a0_v2 = bitcast <2 x half> %op_1_a0_load to <2 x half>
  %op_1_a0_sram_e0 = extractelement <2 x half> %op_1_a0_v2, i32 0
  %op_1_a0_sram_e1 = extractelement <2 x half> %op_1_a0_v2, i32 1
  %op_1_a0_sram_v0 = insertelement <64 x half> undef, half %op_1_a0_sram_e0, i32 0
  %op_1_a0_sram = insertelement <64 x half> %op_1_a0_sram_v0, half %op_1_a0_sram_e1, i32 1
  %op_1_b0x0_row = add i32 %morton_x, 0
  %op_1_b0x0_col = add i32 %morton_y, 0
  %op_1_b0x0_r_0 = add i32 %op_1_b0x0_row, 0
  %op_1_b0x0_addr_0 = mul i32 %op_1_b0x0_r_0, 16
  %op_1_b0x0_addr2_0 = add i32 %op_1_b0x0_addr_0, %op_1_b0x0_col
  %op_1_b0x0_byte_0 = mul i32 %op_1_b0x0_addr2_0, 2
  %op_1_b0x0_byte64_0 = zext i32 %op_1_b0x0_byte_0 to i64
  %op_1_b0x0_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x0_byte64_0
  %op_1_b0x0_typed_0 = bitcast i8 addrspace(3)* %op_1_b0x0_ptr_0 to half addrspace(3)*
  %op_1_b0x0_load_0 = load half, half addrspace(3)* %op_1_b0x0_typed_0
  %op_1_b0x0_r_1 = add i32 %op_1_b0x0_row, 1
  %op_1_b0x0_addr_1 = mul i32 %op_1_b0x0_r_1, 16
  %op_1_b0x0_addr2_1 = add i32 %op_1_b0x0_addr_1, %op_1_b0x0_col
  %op_1_b0x0_byte_1 = mul i32 %op_1_b0x0_addr2_1, 2
  %op_1_b0x0_byte64_1 = zext i32 %op_1_b0x0_byte_1 to i64
  %op_1_b0x0_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x0_byte64_1
  %op_1_b0x0_typed_1 = bitcast i8 addrspace(3)* %op_1_b0x0_ptr_1 to half addrspace(3)*
  %op_1_b0x0_load_1 = load half, half addrspace(3)* %op_1_b0x0_typed_1
  %op_1_b0x0_v2_a = insertelement <2 x half> undef, half %op_1_b0x0_load_0, i32 0
  %op_1_b0x0_v2 = insertelement <2 x half> %op_1_b0x0_v2_a, half %op_1_b0x0_load_1, i32 1
  %op_1_b0x0_sram_e0 = extractelement <2 x half> %op_1_b0x0_v2, i32 0
  %op_1_b0x0_sram_e1 = extractelement <2 x half> %op_1_b0x0_v2, i32 1
  %op_1_b0x0_sram_v0 = insertelement <64 x half> undef, half %op_1_b0x0_sram_e0, i32 0
  %op_1_b0x0_sram = insertelement <64 x half> %op_1_b0x0_sram_v0, half %op_1_b0x0_sram_e1, i32 1
  %op_1_c0x0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a0_sram, <64 x half> %op_1_b0x0_sram, <64 x float> %op_0_c1x0) #3
  %op_1_b0x1_row = add i32 %morton_x, 8
  %op_1_b0x1_col = add i32 %morton_y, 0
  %op_1_b0x1_r_0 = add i32 %op_1_b0x1_row, 0
  %op_1_b0x1_addr_0 = mul i32 %op_1_b0x1_r_0, 16
  %op_1_b0x1_addr2_0 = add i32 %op_1_b0x1_addr_0, %op_1_b0x1_col
  %op_1_b0x1_byte_0 = mul i32 %op_1_b0x1_addr2_0, 2
  %op_1_b0x1_byte64_0 = zext i32 %op_1_b0x1_byte_0 to i64
  %op_1_b0x1_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x1_byte64_0
  %op_1_b0x1_typed_0 = bitcast i8 addrspace(3)* %op_1_b0x1_ptr_0 to half addrspace(3)*
  %op_1_b0x1_load_0 = load half, half addrspace(3)* %op_1_b0x1_typed_0
  %op_1_b0x1_r_1 = add i32 %op_1_b0x1_row, 1
  %op_1_b0x1_addr_1 = mul i32 %op_1_b0x1_r_1, 16
  %op_1_b0x1_addr2_1 = add i32 %op_1_b0x1_addr_1, %op_1_b0x1_col
  %op_1_b0x1_byte_1 = mul i32 %op_1_b0x1_addr2_1, 2
  %op_1_b0x1_byte64_1 = zext i32 %op_1_b0x1_byte_1 to i64
  %op_1_b0x1_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x1_byte64_1
  %op_1_b0x1_typed_1 = bitcast i8 addrspace(3)* %op_1_b0x1_ptr_1 to half addrspace(3)*
  %op_1_b0x1_load_1 = load half, half addrspace(3)* %op_1_b0x1_typed_1
  %op_1_b0x1_v2_a = insertelement <2 x half> undef, half %op_1_b0x1_load_0, i32 0
  %op_1_b0x1_v2 = insertelement <2 x half> %op_1_b0x1_v2_a, half %op_1_b0x1_load_1, i32 1
  %op_1_b0x1_sram_e0 = extractelement <2 x half> %op_1_b0x1_v2, i32 0
  %op_1_b0x1_sram_e1 = extractelement <2 x half> %op_1_b0x1_v2, i32 1
  %op_1_b0x1_sram_v0 = insertelement <64 x half> undef, half %op_1_b0x1_sram_e0, i32 0
  %op_1_b0x1_sram = insertelement <64 x half> %op_1_b0x1_sram_v0, half %op_1_b0x1_sram_e1, i32 1
  %op_1_c0x1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a0_sram, <64 x half> %op_1_b0x1_sram, <64 x float> %op_0_c1x1) #3
  %op_1_b0x2_row = add i32 %morton_x, 16
  %op_1_b0x2_col = add i32 %morton_y, 0
  %op_1_b0x2_r_0 = add i32 %op_1_b0x2_row, 0
  %op_1_b0x2_addr_0 = mul i32 %op_1_b0x2_r_0, 16
  %op_1_b0x2_addr2_0 = add i32 %op_1_b0x2_addr_0, %op_1_b0x2_col
  %op_1_b0x2_byte_0 = mul i32 %op_1_b0x2_addr2_0, 2
  %op_1_b0x2_byte64_0 = zext i32 %op_1_b0x2_byte_0 to i64
  %op_1_b0x2_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x2_byte64_0
  %op_1_b0x2_typed_0 = bitcast i8 addrspace(3)* %op_1_b0x2_ptr_0 to half addrspace(3)*
  %op_1_b0x2_load_0 = load half, half addrspace(3)* %op_1_b0x2_typed_0
  %op_1_b0x2_r_1 = add i32 %op_1_b0x2_row, 1
  %op_1_b0x2_addr_1 = mul i32 %op_1_b0x2_r_1, 16
  %op_1_b0x2_addr2_1 = add i32 %op_1_b0x2_addr_1, %op_1_b0x2_col
  %op_1_b0x2_byte_1 = mul i32 %op_1_b0x2_addr2_1, 2
  %op_1_b0x2_byte64_1 = zext i32 %op_1_b0x2_byte_1 to i64
  %op_1_b0x2_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x2_byte64_1
  %op_1_b0x2_typed_1 = bitcast i8 addrspace(3)* %op_1_b0x2_ptr_1 to half addrspace(3)*
  %op_1_b0x2_load_1 = load half, half addrspace(3)* %op_1_b0x2_typed_1
  %op_1_b0x2_v2_a = insertelement <2 x half> undef, half %op_1_b0x2_load_0, i32 0
  %op_1_b0x2_v2 = insertelement <2 x half> %op_1_b0x2_v2_a, half %op_1_b0x2_load_1, i32 1
  %op_1_b0x2_sram_e0 = extractelement <2 x half> %op_1_b0x2_v2, i32 0
  %op_1_b0x2_sram_e1 = extractelement <2 x half> %op_1_b0x2_v2, i32 1
  %op_1_b0x2_sram_v0 = insertelement <64 x half> undef, half %op_1_b0x2_sram_e0, i32 0
  %op_1_b0x2_sram = insertelement <64 x half> %op_1_b0x2_sram_v0, half %op_1_b0x2_sram_e1, i32 1
  %op_1_c0x2 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a0_sram, <64 x half> %op_1_b0x2_sram, <64 x float> %op_0_c1x2) #3
  %op_1_b0x3_row = add i32 %morton_x, 24
  %op_1_b0x3_col = add i32 %morton_y, 0
  %op_1_b0x3_r_0 = add i32 %op_1_b0x3_row, 0
  %op_1_b0x3_addr_0 = mul i32 %op_1_b0x3_r_0, 16
  %op_1_b0x3_addr2_0 = add i32 %op_1_b0x3_addr_0, %op_1_b0x3_col
  %op_1_b0x3_byte_0 = mul i32 %op_1_b0x3_addr2_0, 2
  %op_1_b0x3_byte64_0 = zext i32 %op_1_b0x3_byte_0 to i64
  %op_1_b0x3_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x3_byte64_0
  %op_1_b0x3_typed_0 = bitcast i8 addrspace(3)* %op_1_b0x3_ptr_0 to half addrspace(3)*
  %op_1_b0x3_load_0 = load half, half addrspace(3)* %op_1_b0x3_typed_0
  %op_1_b0x3_r_1 = add i32 %op_1_b0x3_row, 1
  %op_1_b0x3_addr_1 = mul i32 %op_1_b0x3_r_1, 16
  %op_1_b0x3_addr2_1 = add i32 %op_1_b0x3_addr_1, %op_1_b0x3_col
  %op_1_b0x3_byte_1 = mul i32 %op_1_b0x3_addr2_1, 2
  %op_1_b0x3_byte64_1 = zext i32 %op_1_b0x3_byte_1 to i64
  %op_1_b0x3_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x3_byte64_1
  %op_1_b0x3_typed_1 = bitcast i8 addrspace(3)* %op_1_b0x3_ptr_1 to half addrspace(3)*
  %op_1_b0x3_load_1 = load half, half addrspace(3)* %op_1_b0x3_typed_1
  %op_1_b0x3_v2_a = insertelement <2 x half> undef, half %op_1_b0x3_load_0, i32 0
  %op_1_b0x3_v2 = insertelement <2 x half> %op_1_b0x3_v2_a, half %op_1_b0x3_load_1, i32 1
  %op_1_b0x3_sram_e0 = extractelement <2 x half> %op_1_b0x3_v2, i32 0
  %op_1_b0x3_sram_e1 = extractelement <2 x half> %op_1_b0x3_v2, i32 1
  %op_1_b0x3_sram_v0 = insertelement <64 x half> undef, half %op_1_b0x3_sram_e0, i32 0
  %op_1_b0x3_sram = insertelement <64 x half> %op_1_b0x3_sram_v0, half %op_1_b0x3_sram_e1, i32 1
  %op_1_c0x3 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a0_sram, <64 x half> %op_1_b0x3_sram, <64 x float> %op_0_c1x3) #3
  %op_1_b0x4_row = add i32 %morton_x, 32
  %op_1_b0x4_col = add i32 %morton_y, 0
  %op_1_b0x4_r_0 = add i32 %op_1_b0x4_row, 0
  %op_1_b0x4_addr_0 = mul i32 %op_1_b0x4_r_0, 16
  %op_1_b0x4_addr2_0 = add i32 %op_1_b0x4_addr_0, %op_1_b0x4_col
  %op_1_b0x4_byte_0 = mul i32 %op_1_b0x4_addr2_0, 2
  %op_1_b0x4_byte64_0 = zext i32 %op_1_b0x4_byte_0 to i64
  %op_1_b0x4_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x4_byte64_0
  %op_1_b0x4_typed_0 = bitcast i8 addrspace(3)* %op_1_b0x4_ptr_0 to half addrspace(3)*
  %op_1_b0x4_load_0 = load half, half addrspace(3)* %op_1_b0x4_typed_0
  %op_1_b0x4_r_1 = add i32 %op_1_b0x4_row, 1
  %op_1_b0x4_addr_1 = mul i32 %op_1_b0x4_r_1, 16
  %op_1_b0x4_addr2_1 = add i32 %op_1_b0x4_addr_1, %op_1_b0x4_col
  %op_1_b0x4_byte_1 = mul i32 %op_1_b0x4_addr2_1, 2
  %op_1_b0x4_byte64_1 = zext i32 %op_1_b0x4_byte_1 to i64
  %op_1_b0x4_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x4_byte64_1
  %op_1_b0x4_typed_1 = bitcast i8 addrspace(3)* %op_1_b0x4_ptr_1 to half addrspace(3)*
  %op_1_b0x4_load_1 = load half, half addrspace(3)* %op_1_b0x4_typed_1
  %op_1_b0x4_v2_a = insertelement <2 x half> undef, half %op_1_b0x4_load_0, i32 0
  %op_1_b0x4_v2 = insertelement <2 x half> %op_1_b0x4_v2_a, half %op_1_b0x4_load_1, i32 1
  %op_1_b0x4_sram_e0 = extractelement <2 x half> %op_1_b0x4_v2, i32 0
  %op_1_b0x4_sram_e1 = extractelement <2 x half> %op_1_b0x4_v2, i32 1
  %op_1_b0x4_sram_v0 = insertelement <64 x half> undef, half %op_1_b0x4_sram_e0, i32 0
  %op_1_b0x4_sram = insertelement <64 x half> %op_1_b0x4_sram_v0, half %op_1_b0x4_sram_e1, i32 1
  %op_1_c0x4 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a0_sram, <64 x half> %op_1_b0x4_sram, <64 x float> %op_0_c1x4) #3
  %op_1_b0x5_row = add i32 %morton_x, 40
  %op_1_b0x5_col = add i32 %morton_y, 0
  %op_1_b0x5_r_0 = add i32 %op_1_b0x5_row, 0
  %op_1_b0x5_addr_0 = mul i32 %op_1_b0x5_r_0, 16
  %op_1_b0x5_addr2_0 = add i32 %op_1_b0x5_addr_0, %op_1_b0x5_col
  %op_1_b0x5_byte_0 = mul i32 %op_1_b0x5_addr2_0, 2
  %op_1_b0x5_byte64_0 = zext i32 %op_1_b0x5_byte_0 to i64
  %op_1_b0x5_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x5_byte64_0
  %op_1_b0x5_typed_0 = bitcast i8 addrspace(3)* %op_1_b0x5_ptr_0 to half addrspace(3)*
  %op_1_b0x5_load_0 = load half, half addrspace(3)* %op_1_b0x5_typed_0
  %op_1_b0x5_r_1 = add i32 %op_1_b0x5_row, 1
  %op_1_b0x5_addr_1 = mul i32 %op_1_b0x5_r_1, 16
  %op_1_b0x5_addr2_1 = add i32 %op_1_b0x5_addr_1, %op_1_b0x5_col
  %op_1_b0x5_byte_1 = mul i32 %op_1_b0x5_addr2_1, 2
  %op_1_b0x5_byte64_1 = zext i32 %op_1_b0x5_byte_1 to i64
  %op_1_b0x5_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x5_byte64_1
  %op_1_b0x5_typed_1 = bitcast i8 addrspace(3)* %op_1_b0x5_ptr_1 to half addrspace(3)*
  %op_1_b0x5_load_1 = load half, half addrspace(3)* %op_1_b0x5_typed_1
  %op_1_b0x5_v2_a = insertelement <2 x half> undef, half %op_1_b0x5_load_0, i32 0
  %op_1_b0x5_v2 = insertelement <2 x half> %op_1_b0x5_v2_a, half %op_1_b0x5_load_1, i32 1
  %op_1_b0x5_sram_e0 = extractelement <2 x half> %op_1_b0x5_v2, i32 0
  %op_1_b0x5_sram_e1 = extractelement <2 x half> %op_1_b0x5_v2, i32 1
  %op_1_b0x5_sram_v0 = insertelement <64 x half> undef, half %op_1_b0x5_sram_e0, i32 0
  %op_1_b0x5_sram = insertelement <64 x half> %op_1_b0x5_sram_v0, half %op_1_b0x5_sram_e1, i32 1
  %op_1_c0x5 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a0_sram, <64 x half> %op_1_b0x5_sram, <64 x float> %op_0_c1x5) #3
  %op_1_b0x6_row = add i32 %morton_x, 48
  %op_1_b0x6_col = add i32 %morton_y, 0
  %op_1_b0x6_r_0 = add i32 %op_1_b0x6_row, 0
  %op_1_b0x6_addr_0 = mul i32 %op_1_b0x6_r_0, 16
  %op_1_b0x6_addr2_0 = add i32 %op_1_b0x6_addr_0, %op_1_b0x6_col
  %op_1_b0x6_byte_0 = mul i32 %op_1_b0x6_addr2_0, 2
  %op_1_b0x6_byte64_0 = zext i32 %op_1_b0x6_byte_0 to i64
  %op_1_b0x6_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x6_byte64_0
  %op_1_b0x6_typed_0 = bitcast i8 addrspace(3)* %op_1_b0x6_ptr_0 to half addrspace(3)*
  %op_1_b0x6_load_0 = load half, half addrspace(3)* %op_1_b0x6_typed_0
  %op_1_b0x6_r_1 = add i32 %op_1_b0x6_row, 1
  %op_1_b0x6_addr_1 = mul i32 %op_1_b0x6_r_1, 16
  %op_1_b0x6_addr2_1 = add i32 %op_1_b0x6_addr_1, %op_1_b0x6_col
  %op_1_b0x6_byte_1 = mul i32 %op_1_b0x6_addr2_1, 2
  %op_1_b0x6_byte64_1 = zext i32 %op_1_b0x6_byte_1 to i64
  %op_1_b0x6_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x6_byte64_1
  %op_1_b0x6_typed_1 = bitcast i8 addrspace(3)* %op_1_b0x6_ptr_1 to half addrspace(3)*
  %op_1_b0x6_load_1 = load half, half addrspace(3)* %op_1_b0x6_typed_1
  %op_1_b0x6_v2_a = insertelement <2 x half> undef, half %op_1_b0x6_load_0, i32 0
  %op_1_b0x6_v2 = insertelement <2 x half> %op_1_b0x6_v2_a, half %op_1_b0x6_load_1, i32 1
  %op_1_b0x6_sram_e0 = extractelement <2 x half> %op_1_b0x6_v2, i32 0
  %op_1_b0x6_sram_e1 = extractelement <2 x half> %op_1_b0x6_v2, i32 1
  %op_1_b0x6_sram_v0 = insertelement <64 x half> undef, half %op_1_b0x6_sram_e0, i32 0
  %op_1_b0x6_sram = insertelement <64 x half> %op_1_b0x6_sram_v0, half %op_1_b0x6_sram_e1, i32 1
  %op_1_c0x6 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a0_sram, <64 x half> %op_1_b0x6_sram, <64 x float> %op_0_c1x6) #3
  %op_1_b0x7_row = add i32 %morton_x, 56
  %op_1_b0x7_col = add i32 %morton_y, 0
  %op_1_b0x7_r_0 = add i32 %op_1_b0x7_row, 0
  %op_1_b0x7_addr_0 = mul i32 %op_1_b0x7_r_0, 16
  %op_1_b0x7_addr2_0 = add i32 %op_1_b0x7_addr_0, %op_1_b0x7_col
  %op_1_b0x7_byte_0 = mul i32 %op_1_b0x7_addr2_0, 2
  %op_1_b0x7_byte64_0 = zext i32 %op_1_b0x7_byte_0 to i64
  %op_1_b0x7_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x7_byte64_0
  %op_1_b0x7_typed_0 = bitcast i8 addrspace(3)* %op_1_b0x7_ptr_0 to half addrspace(3)*
  %op_1_b0x7_load_0 = load half, half addrspace(3)* %op_1_b0x7_typed_0
  %op_1_b0x7_r_1 = add i32 %op_1_b0x7_row, 1
  %op_1_b0x7_addr_1 = mul i32 %op_1_b0x7_r_1, 16
  %op_1_b0x7_addr2_1 = add i32 %op_1_b0x7_addr_1, %op_1_b0x7_col
  %op_1_b0x7_byte_1 = mul i32 %op_1_b0x7_addr2_1, 2
  %op_1_b0x7_byte64_1 = zext i32 %op_1_b0x7_byte_1 to i64
  %op_1_b0x7_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b0x7_byte64_1
  %op_1_b0x7_typed_1 = bitcast i8 addrspace(3)* %op_1_b0x7_ptr_1 to half addrspace(3)*
  %op_1_b0x7_load_1 = load half, half addrspace(3)* %op_1_b0x7_typed_1
  %op_1_b0x7_v2_a = insertelement <2 x half> undef, half %op_1_b0x7_load_0, i32 0
  %op_1_b0x7_v2 = insertelement <2 x half> %op_1_b0x7_v2_a, half %op_1_b0x7_load_1, i32 1
  %op_1_b0x7_sram_e0 = extractelement <2 x half> %op_1_b0x7_v2, i32 0
  %op_1_b0x7_sram_e1 = extractelement <2 x half> %op_1_b0x7_v2, i32 1
  %op_1_b0x7_sram_v0 = insertelement <64 x half> undef, half %op_1_b0x7_sram_e0, i32 0
  %op_1_b0x7_sram = insertelement <64 x half> %op_1_b0x7_sram_v0, half %op_1_b0x7_sram_e1, i32 1
  %op_1_c0x7 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a0_sram, <64 x half> %op_1_b0x7_sram, <64 x float> %op_0_c1x7) #3
  %op_1_a1_seq = add i32 %clamped_par_off, 0
  %op_1_a1_head = add i32 %morton_x, 24
  %op_1_a1_addr = mul i32 %op_1_a1_seq, 64
  %op_1_a1_addr2 = add i32 %op_1_a1_addr, %op_1_a1_head
  %op_1_a1_byte = mul i32 %op_1_a1_addr2, 2
  %op_1_a1_byte64 = zext i32 %op_1_a1_byte to i64
  %op_1_a1_ptr = getelementptr i8, i8 addrspace(1)* %Q, i64 %op_1_a1_byte64
  %op_1_a1_typed = bitcast i8 addrspace(1)* %op_1_a1_ptr to <2 x half> addrspace(1)*
  %op_1_a1_load = load <2 x half>, <2 x half> addrspace(1)* %op_1_a1_typed, align 4
  %op_1_a1_v2 = bitcast <2 x half> %op_1_a1_load to <2 x half>
  %op_1_a1_sram_e0 = extractelement <2 x half> %op_1_a1_v2, i32 0
  %op_1_a1_sram_e1 = extractelement <2 x half> %op_1_a1_v2, i32 1
  %op_1_a1_sram_v0 = insertelement <64 x half> undef, half %op_1_a1_sram_e0, i32 0
  %op_1_a1_sram = insertelement <64 x half> %op_1_a1_sram_v0, half %op_1_a1_sram_e1, i32 1
  %op_1_b1x0_row = add i32 %morton_x, 0
  %op_1_b1x0_col = add i32 %morton_y, 8
  %op_1_b1x0_r_0 = add i32 %op_1_b1x0_row, 0
  %op_1_b1x0_addr_0 = mul i32 %op_1_b1x0_r_0, 16
  %op_1_b1x0_addr2_0 = add i32 %op_1_b1x0_addr_0, %op_1_b1x0_col
  %op_1_b1x0_byte_0 = mul i32 %op_1_b1x0_addr2_0, 2
  %op_1_b1x0_byte64_0 = zext i32 %op_1_b1x0_byte_0 to i64
  %op_1_b1x0_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x0_byte64_0
  %op_1_b1x0_typed_0 = bitcast i8 addrspace(3)* %op_1_b1x0_ptr_0 to half addrspace(3)*
  %op_1_b1x0_load_0 = load half, half addrspace(3)* %op_1_b1x0_typed_0
  %op_1_b1x0_r_1 = add i32 %op_1_b1x0_row, 1
  %op_1_b1x0_addr_1 = mul i32 %op_1_b1x0_r_1, 16
  %op_1_b1x0_addr2_1 = add i32 %op_1_b1x0_addr_1, %op_1_b1x0_col
  %op_1_b1x0_byte_1 = mul i32 %op_1_b1x0_addr2_1, 2
  %op_1_b1x0_byte64_1 = zext i32 %op_1_b1x0_byte_1 to i64
  %op_1_b1x0_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x0_byte64_1
  %op_1_b1x0_typed_1 = bitcast i8 addrspace(3)* %op_1_b1x0_ptr_1 to half addrspace(3)*
  %op_1_b1x0_load_1 = load half, half addrspace(3)* %op_1_b1x0_typed_1
  %op_1_b1x0_v2_a = insertelement <2 x half> undef, half %op_1_b1x0_load_0, i32 0
  %op_1_b1x0_v2 = insertelement <2 x half> %op_1_b1x0_v2_a, half %op_1_b1x0_load_1, i32 1
  %op_1_b1x0_sram_e0 = extractelement <2 x half> %op_1_b1x0_v2, i32 0
  %op_1_b1x0_sram_e1 = extractelement <2 x half> %op_1_b1x0_v2, i32 1
  %op_1_b1x0_sram_v0 = insertelement <64 x half> undef, half %op_1_b1x0_sram_e0, i32 0
  %op_1_b1x0_sram = insertelement <64 x half> %op_1_b1x0_sram_v0, half %op_1_b1x0_sram_e1, i32 1
  %op_1_c1x0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a1_sram, <64 x half> %op_1_b1x0_sram, <64 x float> %op_1_c0x0) #3
  %op_1_b1x1_row = add i32 %morton_x, 8
  %op_1_b1x1_col = add i32 %morton_y, 8
  %op_1_b1x1_r_0 = add i32 %op_1_b1x1_row, 0
  %op_1_b1x1_addr_0 = mul i32 %op_1_b1x1_r_0, 16
  %op_1_b1x1_addr2_0 = add i32 %op_1_b1x1_addr_0, %op_1_b1x1_col
  %op_1_b1x1_byte_0 = mul i32 %op_1_b1x1_addr2_0, 2
  %op_1_b1x1_byte64_0 = zext i32 %op_1_b1x1_byte_0 to i64
  %op_1_b1x1_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x1_byte64_0
  %op_1_b1x1_typed_0 = bitcast i8 addrspace(3)* %op_1_b1x1_ptr_0 to half addrspace(3)*
  %op_1_b1x1_load_0 = load half, half addrspace(3)* %op_1_b1x1_typed_0
  %op_1_b1x1_r_1 = add i32 %op_1_b1x1_row, 1
  %op_1_b1x1_addr_1 = mul i32 %op_1_b1x1_r_1, 16
  %op_1_b1x1_addr2_1 = add i32 %op_1_b1x1_addr_1, %op_1_b1x1_col
  %op_1_b1x1_byte_1 = mul i32 %op_1_b1x1_addr2_1, 2
  %op_1_b1x1_byte64_1 = zext i32 %op_1_b1x1_byte_1 to i64
  %op_1_b1x1_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x1_byte64_1
  %op_1_b1x1_typed_1 = bitcast i8 addrspace(3)* %op_1_b1x1_ptr_1 to half addrspace(3)*
  %op_1_b1x1_load_1 = load half, half addrspace(3)* %op_1_b1x1_typed_1
  %op_1_b1x1_v2_a = insertelement <2 x half> undef, half %op_1_b1x1_load_0, i32 0
  %op_1_b1x1_v2 = insertelement <2 x half> %op_1_b1x1_v2_a, half %op_1_b1x1_load_1, i32 1
  %op_1_b1x1_sram_e0 = extractelement <2 x half> %op_1_b1x1_v2, i32 0
  %op_1_b1x1_sram_e1 = extractelement <2 x half> %op_1_b1x1_v2, i32 1
  %op_1_b1x1_sram_v0 = insertelement <64 x half> undef, half %op_1_b1x1_sram_e0, i32 0
  %op_1_b1x1_sram = insertelement <64 x half> %op_1_b1x1_sram_v0, half %op_1_b1x1_sram_e1, i32 1
  %op_1_c1x1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a1_sram, <64 x half> %op_1_b1x1_sram, <64 x float> %op_1_c0x1) #3
  %op_1_b1x2_row = add i32 %morton_x, 16
  %op_1_b1x2_col = add i32 %morton_y, 8
  %op_1_b1x2_r_0 = add i32 %op_1_b1x2_row, 0
  %op_1_b1x2_addr_0 = mul i32 %op_1_b1x2_r_0, 16
  %op_1_b1x2_addr2_0 = add i32 %op_1_b1x2_addr_0, %op_1_b1x2_col
  %op_1_b1x2_byte_0 = mul i32 %op_1_b1x2_addr2_0, 2
  %op_1_b1x2_byte64_0 = zext i32 %op_1_b1x2_byte_0 to i64
  %op_1_b1x2_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x2_byte64_0
  %op_1_b1x2_typed_0 = bitcast i8 addrspace(3)* %op_1_b1x2_ptr_0 to half addrspace(3)*
  %op_1_b1x2_load_0 = load half, half addrspace(3)* %op_1_b1x2_typed_0
  %op_1_b1x2_r_1 = add i32 %op_1_b1x2_row, 1
  %op_1_b1x2_addr_1 = mul i32 %op_1_b1x2_r_1, 16
  %op_1_b1x2_addr2_1 = add i32 %op_1_b1x2_addr_1, %op_1_b1x2_col
  %op_1_b1x2_byte_1 = mul i32 %op_1_b1x2_addr2_1, 2
  %op_1_b1x2_byte64_1 = zext i32 %op_1_b1x2_byte_1 to i64
  %op_1_b1x2_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x2_byte64_1
  %op_1_b1x2_typed_1 = bitcast i8 addrspace(3)* %op_1_b1x2_ptr_1 to half addrspace(3)*
  %op_1_b1x2_load_1 = load half, half addrspace(3)* %op_1_b1x2_typed_1
  %op_1_b1x2_v2_a = insertelement <2 x half> undef, half %op_1_b1x2_load_0, i32 0
  %op_1_b1x2_v2 = insertelement <2 x half> %op_1_b1x2_v2_a, half %op_1_b1x2_load_1, i32 1
  %op_1_b1x2_sram_e0 = extractelement <2 x half> %op_1_b1x2_v2, i32 0
  %op_1_b1x2_sram_e1 = extractelement <2 x half> %op_1_b1x2_v2, i32 1
  %op_1_b1x2_sram_v0 = insertelement <64 x half> undef, half %op_1_b1x2_sram_e0, i32 0
  %op_1_b1x2_sram = insertelement <64 x half> %op_1_b1x2_sram_v0, half %op_1_b1x2_sram_e1, i32 1
  %op_1_c1x2 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a1_sram, <64 x half> %op_1_b1x2_sram, <64 x float> %op_1_c0x2) #3
  %op_1_b1x3_row = add i32 %morton_x, 24
  %op_1_b1x3_col = add i32 %morton_y, 8
  %op_1_b1x3_r_0 = add i32 %op_1_b1x3_row, 0
  %op_1_b1x3_addr_0 = mul i32 %op_1_b1x3_r_0, 16
  %op_1_b1x3_addr2_0 = add i32 %op_1_b1x3_addr_0, %op_1_b1x3_col
  %op_1_b1x3_byte_0 = mul i32 %op_1_b1x3_addr2_0, 2
  %op_1_b1x3_byte64_0 = zext i32 %op_1_b1x3_byte_0 to i64
  %op_1_b1x3_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x3_byte64_0
  %op_1_b1x3_typed_0 = bitcast i8 addrspace(3)* %op_1_b1x3_ptr_0 to half addrspace(3)*
  %op_1_b1x3_load_0 = load half, half addrspace(3)* %op_1_b1x3_typed_0
  %op_1_b1x3_r_1 = add i32 %op_1_b1x3_row, 1
  %op_1_b1x3_addr_1 = mul i32 %op_1_b1x3_r_1, 16
  %op_1_b1x3_addr2_1 = add i32 %op_1_b1x3_addr_1, %op_1_b1x3_col
  %op_1_b1x3_byte_1 = mul i32 %op_1_b1x3_addr2_1, 2
  %op_1_b1x3_byte64_1 = zext i32 %op_1_b1x3_byte_1 to i64
  %op_1_b1x3_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x3_byte64_1
  %op_1_b1x3_typed_1 = bitcast i8 addrspace(3)* %op_1_b1x3_ptr_1 to half addrspace(3)*
  %op_1_b1x3_load_1 = load half, half addrspace(3)* %op_1_b1x3_typed_1
  %op_1_b1x3_v2_a = insertelement <2 x half> undef, half %op_1_b1x3_load_0, i32 0
  %op_1_b1x3_v2 = insertelement <2 x half> %op_1_b1x3_v2_a, half %op_1_b1x3_load_1, i32 1
  %op_1_b1x3_sram_e0 = extractelement <2 x half> %op_1_b1x3_v2, i32 0
  %op_1_b1x3_sram_e1 = extractelement <2 x half> %op_1_b1x3_v2, i32 1
  %op_1_b1x3_sram_v0 = insertelement <64 x half> undef, half %op_1_b1x3_sram_e0, i32 0
  %op_1_b1x3_sram = insertelement <64 x half> %op_1_b1x3_sram_v0, half %op_1_b1x3_sram_e1, i32 1
  %op_1_c1x3 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a1_sram, <64 x half> %op_1_b1x3_sram, <64 x float> %op_1_c0x3) #3
  %op_1_b1x4_row = add i32 %morton_x, 32
  %op_1_b1x4_col = add i32 %morton_y, 8
  %op_1_b1x4_r_0 = add i32 %op_1_b1x4_row, 0
  %op_1_b1x4_addr_0 = mul i32 %op_1_b1x4_r_0, 16
  %op_1_b1x4_addr2_0 = add i32 %op_1_b1x4_addr_0, %op_1_b1x4_col
  %op_1_b1x4_byte_0 = mul i32 %op_1_b1x4_addr2_0, 2
  %op_1_b1x4_byte64_0 = zext i32 %op_1_b1x4_byte_0 to i64
  %op_1_b1x4_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x4_byte64_0
  %op_1_b1x4_typed_0 = bitcast i8 addrspace(3)* %op_1_b1x4_ptr_0 to half addrspace(3)*
  %op_1_b1x4_load_0 = load half, half addrspace(3)* %op_1_b1x4_typed_0
  %op_1_b1x4_r_1 = add i32 %op_1_b1x4_row, 1
  %op_1_b1x4_addr_1 = mul i32 %op_1_b1x4_r_1, 16
  %op_1_b1x4_addr2_1 = add i32 %op_1_b1x4_addr_1, %op_1_b1x4_col
  %op_1_b1x4_byte_1 = mul i32 %op_1_b1x4_addr2_1, 2
  %op_1_b1x4_byte64_1 = zext i32 %op_1_b1x4_byte_1 to i64
  %op_1_b1x4_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x4_byte64_1
  %op_1_b1x4_typed_1 = bitcast i8 addrspace(3)* %op_1_b1x4_ptr_1 to half addrspace(3)*
  %op_1_b1x4_load_1 = load half, half addrspace(3)* %op_1_b1x4_typed_1
  %op_1_b1x4_v2_a = insertelement <2 x half> undef, half %op_1_b1x4_load_0, i32 0
  %op_1_b1x4_v2 = insertelement <2 x half> %op_1_b1x4_v2_a, half %op_1_b1x4_load_1, i32 1
  %op_1_b1x4_sram_e0 = extractelement <2 x half> %op_1_b1x4_v2, i32 0
  %op_1_b1x4_sram_e1 = extractelement <2 x half> %op_1_b1x4_v2, i32 1
  %op_1_b1x4_sram_v0 = insertelement <64 x half> undef, half %op_1_b1x4_sram_e0, i32 0
  %op_1_b1x4_sram = insertelement <64 x half> %op_1_b1x4_sram_v0, half %op_1_b1x4_sram_e1, i32 1
  %op_1_c1x4 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a1_sram, <64 x half> %op_1_b1x4_sram, <64 x float> %op_1_c0x4) #3
  %op_1_b1x5_row = add i32 %morton_x, 40
  %op_1_b1x5_col = add i32 %morton_y, 8
  %op_1_b1x5_r_0 = add i32 %op_1_b1x5_row, 0
  %op_1_b1x5_addr_0 = mul i32 %op_1_b1x5_r_0, 16
  %op_1_b1x5_addr2_0 = add i32 %op_1_b1x5_addr_0, %op_1_b1x5_col
  %op_1_b1x5_byte_0 = mul i32 %op_1_b1x5_addr2_0, 2
  %op_1_b1x5_byte64_0 = zext i32 %op_1_b1x5_byte_0 to i64
  %op_1_b1x5_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x5_byte64_0
  %op_1_b1x5_typed_0 = bitcast i8 addrspace(3)* %op_1_b1x5_ptr_0 to half addrspace(3)*
  %op_1_b1x5_load_0 = load half, half addrspace(3)* %op_1_b1x5_typed_0
  %op_1_b1x5_r_1 = add i32 %op_1_b1x5_row, 1
  %op_1_b1x5_addr_1 = mul i32 %op_1_b1x5_r_1, 16
  %op_1_b1x5_addr2_1 = add i32 %op_1_b1x5_addr_1, %op_1_b1x5_col
  %op_1_b1x5_byte_1 = mul i32 %op_1_b1x5_addr2_1, 2
  %op_1_b1x5_byte64_1 = zext i32 %op_1_b1x5_byte_1 to i64
  %op_1_b1x5_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x5_byte64_1
  %op_1_b1x5_typed_1 = bitcast i8 addrspace(3)* %op_1_b1x5_ptr_1 to half addrspace(3)*
  %op_1_b1x5_load_1 = load half, half addrspace(3)* %op_1_b1x5_typed_1
  %op_1_b1x5_v2_a = insertelement <2 x half> undef, half %op_1_b1x5_load_0, i32 0
  %op_1_b1x5_v2 = insertelement <2 x half> %op_1_b1x5_v2_a, half %op_1_b1x5_load_1, i32 1
  %op_1_b1x5_sram_e0 = extractelement <2 x half> %op_1_b1x5_v2, i32 0
  %op_1_b1x5_sram_e1 = extractelement <2 x half> %op_1_b1x5_v2, i32 1
  %op_1_b1x5_sram_v0 = insertelement <64 x half> undef, half %op_1_b1x5_sram_e0, i32 0
  %op_1_b1x5_sram = insertelement <64 x half> %op_1_b1x5_sram_v0, half %op_1_b1x5_sram_e1, i32 1
  %op_1_c1x5 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a1_sram, <64 x half> %op_1_b1x5_sram, <64 x float> %op_1_c0x5) #3
  %op_1_b1x6_row = add i32 %morton_x, 48
  %op_1_b1x6_col = add i32 %morton_y, 8
  %op_1_b1x6_r_0 = add i32 %op_1_b1x6_row, 0
  %op_1_b1x6_addr_0 = mul i32 %op_1_b1x6_r_0, 16
  %op_1_b1x6_addr2_0 = add i32 %op_1_b1x6_addr_0, %op_1_b1x6_col
  %op_1_b1x6_byte_0 = mul i32 %op_1_b1x6_addr2_0, 2
  %op_1_b1x6_byte64_0 = zext i32 %op_1_b1x6_byte_0 to i64
  %op_1_b1x6_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x6_byte64_0
  %op_1_b1x6_typed_0 = bitcast i8 addrspace(3)* %op_1_b1x6_ptr_0 to half addrspace(3)*
  %op_1_b1x6_load_0 = load half, half addrspace(3)* %op_1_b1x6_typed_0
  %op_1_b1x6_r_1 = add i32 %op_1_b1x6_row, 1
  %op_1_b1x6_addr_1 = mul i32 %op_1_b1x6_r_1, 16
  %op_1_b1x6_addr2_1 = add i32 %op_1_b1x6_addr_1, %op_1_b1x6_col
  %op_1_b1x6_byte_1 = mul i32 %op_1_b1x6_addr2_1, 2
  %op_1_b1x6_byte64_1 = zext i32 %op_1_b1x6_byte_1 to i64
  %op_1_b1x6_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x6_byte64_1
  %op_1_b1x6_typed_1 = bitcast i8 addrspace(3)* %op_1_b1x6_ptr_1 to half addrspace(3)*
  %op_1_b1x6_load_1 = load half, half addrspace(3)* %op_1_b1x6_typed_1
  %op_1_b1x6_v2_a = insertelement <2 x half> undef, half %op_1_b1x6_load_0, i32 0
  %op_1_b1x6_v2 = insertelement <2 x half> %op_1_b1x6_v2_a, half %op_1_b1x6_load_1, i32 1
  %op_1_b1x6_sram_e0 = extractelement <2 x half> %op_1_b1x6_v2, i32 0
  %op_1_b1x6_sram_e1 = extractelement <2 x half> %op_1_b1x6_v2, i32 1
  %op_1_b1x6_sram_v0 = insertelement <64 x half> undef, half %op_1_b1x6_sram_e0, i32 0
  %op_1_b1x6_sram = insertelement <64 x half> %op_1_b1x6_sram_v0, half %op_1_b1x6_sram_e1, i32 1
  %op_1_c1x6 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a1_sram, <64 x half> %op_1_b1x6_sram, <64 x float> %op_1_c0x6) #3
  %op_1_b1x7_row = add i32 %morton_x, 56
  %op_1_b1x7_col = add i32 %morton_y, 8
  %op_1_b1x7_r_0 = add i32 %op_1_b1x7_row, 0
  %op_1_b1x7_addr_0 = mul i32 %op_1_b1x7_r_0, 16
  %op_1_b1x7_addr2_0 = add i32 %op_1_b1x7_addr_0, %op_1_b1x7_col
  %op_1_b1x7_byte_0 = mul i32 %op_1_b1x7_addr2_0, 2
  %op_1_b1x7_byte64_0 = zext i32 %op_1_b1x7_byte_0 to i64
  %op_1_b1x7_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x7_byte64_0
  %op_1_b1x7_typed_0 = bitcast i8 addrspace(3)* %op_1_b1x7_ptr_0 to half addrspace(3)*
  %op_1_b1x7_load_0 = load half, half addrspace(3)* %op_1_b1x7_typed_0
  %op_1_b1x7_r_1 = add i32 %op_1_b1x7_row, 1
  %op_1_b1x7_addr_1 = mul i32 %op_1_b1x7_r_1, 16
  %op_1_b1x7_addr2_1 = add i32 %op_1_b1x7_addr_1, %op_1_b1x7_col
  %op_1_b1x7_byte_1 = mul i32 %op_1_b1x7_addr2_1, 2
  %op_1_b1x7_byte64_1 = zext i32 %op_1_b1x7_byte_1 to i64
  %op_1_b1x7_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_1_b1x7_byte64_1
  %op_1_b1x7_typed_1 = bitcast i8 addrspace(3)* %op_1_b1x7_ptr_1 to half addrspace(3)*
  %op_1_b1x7_load_1 = load half, half addrspace(3)* %op_1_b1x7_typed_1
  %op_1_b1x7_v2_a = insertelement <2 x half> undef, half %op_1_b1x7_load_0, i32 0
  %op_1_b1x7_v2 = insertelement <2 x half> %op_1_b1x7_v2_a, half %op_1_b1x7_load_1, i32 1
  %op_1_b1x7_sram_e0 = extractelement <2 x half> %op_1_b1x7_v2, i32 0
  %op_1_b1x7_sram_e1 = extractelement <2 x half> %op_1_b1x7_v2, i32 1
  %op_1_b1x7_sram_v0 = insertelement <64 x half> undef, half %op_1_b1x7_sram_e0, i32 0
  %op_1_b1x7_sram = insertelement <64 x half> %op_1_b1x7_sram_v0, half %op_1_b1x7_sram_e1, i32 1
  %op_1_c1x7 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_1_a1_sram, <64 x half> %op_1_b1x7_sram, <64 x float> %op_1_c0x7) #3
  call void @air.wg.barrier(i32 2, i32 1)

  ; === Sync copy (op_2_b) — all threads cooperative ===
  %op_2_bsc_t0 = mul i32 %sidx, 32
  %op_2_bsc_tid = add i32 %op_2_bsc_t0, %lane_id
  %op_2_bsc_drem = sub i32 64, 32
  %op_2_bsc_dcmp = icmp ult i32 %op_2_bsc_drem, 16
  %op_2_bsc_dsrc = select i1 %op_2_bsc_dcmp, i32 %op_2_bsc_drem, i32 16
  %op_2_bsc_srem = sub i32 64, %c
  %op_2_bsc_scmp = icmp ult i32 %op_2_bsc_srem, 64
  %op_2_bsc_ssrc = select i1 %op_2_bsc_scmp, i32 %op_2_bsc_srem, i32 64
  br label %op_2_bsc_pre

op_2_bsc_pre:
  br label %op_2_bsc_hdr

op_2_bsc_hdr:
  %op_2_bsc_i = phi i32 [%op_2_bsc_tid, %op_2_bsc_pre], [%op_2_bsc_inx, %op_2_bsc_st]
  %op_2_bsc_done = icmp uge i32 %op_2_bsc_i, 1024
  br i1 %op_2_bsc_done, label %op_2_bsc_end, label %op_2_bsc_body

op_2_bsc_body:
  %op_2_bsc_row = lshr i32 %op_2_bsc_i, 4
  %op_2_bsc_col = and i32 %op_2_bsc_i, 15
  %op_2_bsc_rok = icmp ult i32 %op_2_bsc_row, %op_2_bsc_ssrc
  %op_2_bsc_cok = icmp ult i32 %op_2_bsc_col, %op_2_bsc_dsrc
  %op_2_bsc_ib = and i1 %op_2_bsc_rok, %op_2_bsc_cok
  br i1 %op_2_bsc_ib, label %op_2_bsc_ld, label %op_2_bsc_zr

op_2_bsc_ld:
  %op_2_bsc_sr = add i32 %c, %op_2_bsc_row
  %op_2_bsc_sa = mul i32 %op_2_bsc_sr, 64
  %op_2_bsc_sc = add i32 32, %op_2_bsc_col
  %op_2_bsc_sad = add i32 %op_2_bsc_sa, %op_2_bsc_sc
  %op_2_bsc_soff = zext i32 %op_2_bsc_sad to i64
  %op_2_bsc_sbyt = mul i64 %op_2_bsc_soff, 2
  %op_2_bsc_sp = getelementptr i8, i8 addrspace(1)* %K, i64 %op_2_bsc_sbyt
  %op_2_bsc_spt = bitcast i8 addrspace(1)* %op_2_bsc_sp to i16 addrspace(1)*
  %op_2_bsc_lv = load i16, i16 addrspace(1)* %op_2_bsc_spt
  br label %op_2_bsc_st

op_2_bsc_zr:
  br label %op_2_bsc_st

op_2_bsc_st:
  %op_2_bsc_val = phi i16 [%op_2_bsc_lv, %op_2_bsc_ld], [0, %op_2_bsc_zr]
  %op_2_bsc_tr = mul i32 %op_2_bsc_row, 16
  %op_2_bsc_ta = add i32 %op_2_bsc_tr, %op_2_bsc_col
  %op_2_bsc_tb = mul i32 %op_2_bsc_ta, 2
  %op_2_bsc_tb64 = zext i32 %op_2_bsc_tb to i64
  %op_2_bsc_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_bsc_tb64
  %op_2_bsc_tpt = bitcast i8 addrspace(3)* %op_2_bsc_tp to i16 addrspace(3)*
  store i16 %op_2_bsc_val, i16 addrspace(3)* %op_2_bsc_tpt
  %op_2_bsc_inx = add i32 %op_2_bsc_i, 64
  br label %op_2_bsc_hdr

op_2_bsc_end:
  call void @air.wg.barrier(i32 2, i32 1)

  %op_2_a0_seq = add i32 %clamped_par_off, 0
  %op_2_a0_head = add i32 %morton_x, 32
  %op_2_a0_addr = mul i32 %op_2_a0_seq, 64
  %op_2_a0_addr2 = add i32 %op_2_a0_addr, %op_2_a0_head
  %op_2_a0_byte = mul i32 %op_2_a0_addr2, 2
  %op_2_a0_byte64 = zext i32 %op_2_a0_byte to i64
  %op_2_a0_ptr = getelementptr i8, i8 addrspace(1)* %Q, i64 %op_2_a0_byte64
  %op_2_a0_typed = bitcast i8 addrspace(1)* %op_2_a0_ptr to <2 x half> addrspace(1)*
  %op_2_a0_load = load <2 x half>, <2 x half> addrspace(1)* %op_2_a0_typed, align 4
  %op_2_a0_v2 = bitcast <2 x half> %op_2_a0_load to <2 x half>
  %op_2_a0_sram_e0 = extractelement <2 x half> %op_2_a0_v2, i32 0
  %op_2_a0_sram_e1 = extractelement <2 x half> %op_2_a0_v2, i32 1
  %op_2_a0_sram_v0 = insertelement <64 x half> undef, half %op_2_a0_sram_e0, i32 0
  %op_2_a0_sram = insertelement <64 x half> %op_2_a0_sram_v0, half %op_2_a0_sram_e1, i32 1
  %op_2_b0x0_row = add i32 %morton_x, 0
  %op_2_b0x0_col = add i32 %morton_y, 0
  %op_2_b0x0_r_0 = add i32 %op_2_b0x0_row, 0
  %op_2_b0x0_addr_0 = mul i32 %op_2_b0x0_r_0, 16
  %op_2_b0x0_addr2_0 = add i32 %op_2_b0x0_addr_0, %op_2_b0x0_col
  %op_2_b0x0_byte_0 = mul i32 %op_2_b0x0_addr2_0, 2
  %op_2_b0x0_byte64_0 = zext i32 %op_2_b0x0_byte_0 to i64
  %op_2_b0x0_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x0_byte64_0
  %op_2_b0x0_typed_0 = bitcast i8 addrspace(3)* %op_2_b0x0_ptr_0 to half addrspace(3)*
  %op_2_b0x0_load_0 = load half, half addrspace(3)* %op_2_b0x0_typed_0
  %op_2_b0x0_r_1 = add i32 %op_2_b0x0_row, 1
  %op_2_b0x0_addr_1 = mul i32 %op_2_b0x0_r_1, 16
  %op_2_b0x0_addr2_1 = add i32 %op_2_b0x0_addr_1, %op_2_b0x0_col
  %op_2_b0x0_byte_1 = mul i32 %op_2_b0x0_addr2_1, 2
  %op_2_b0x0_byte64_1 = zext i32 %op_2_b0x0_byte_1 to i64
  %op_2_b0x0_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x0_byte64_1
  %op_2_b0x0_typed_1 = bitcast i8 addrspace(3)* %op_2_b0x0_ptr_1 to half addrspace(3)*
  %op_2_b0x0_load_1 = load half, half addrspace(3)* %op_2_b0x0_typed_1
  %op_2_b0x0_v2_a = insertelement <2 x half> undef, half %op_2_b0x0_load_0, i32 0
  %op_2_b0x0_v2 = insertelement <2 x half> %op_2_b0x0_v2_a, half %op_2_b0x0_load_1, i32 1
  %op_2_b0x0_sram_e0 = extractelement <2 x half> %op_2_b0x0_v2, i32 0
  %op_2_b0x0_sram_e1 = extractelement <2 x half> %op_2_b0x0_v2, i32 1
  %op_2_b0x0_sram_v0 = insertelement <64 x half> undef, half %op_2_b0x0_sram_e0, i32 0
  %op_2_b0x0_sram = insertelement <64 x half> %op_2_b0x0_sram_v0, half %op_2_b0x0_sram_e1, i32 1
  %op_2_c0x0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a0_sram, <64 x half> %op_2_b0x0_sram, <64 x float> %op_1_c1x0) #3
  %op_2_b0x1_row = add i32 %morton_x, 8
  %op_2_b0x1_col = add i32 %morton_y, 0
  %op_2_b0x1_r_0 = add i32 %op_2_b0x1_row, 0
  %op_2_b0x1_addr_0 = mul i32 %op_2_b0x1_r_0, 16
  %op_2_b0x1_addr2_0 = add i32 %op_2_b0x1_addr_0, %op_2_b0x1_col
  %op_2_b0x1_byte_0 = mul i32 %op_2_b0x1_addr2_0, 2
  %op_2_b0x1_byte64_0 = zext i32 %op_2_b0x1_byte_0 to i64
  %op_2_b0x1_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x1_byte64_0
  %op_2_b0x1_typed_0 = bitcast i8 addrspace(3)* %op_2_b0x1_ptr_0 to half addrspace(3)*
  %op_2_b0x1_load_0 = load half, half addrspace(3)* %op_2_b0x1_typed_0
  %op_2_b0x1_r_1 = add i32 %op_2_b0x1_row, 1
  %op_2_b0x1_addr_1 = mul i32 %op_2_b0x1_r_1, 16
  %op_2_b0x1_addr2_1 = add i32 %op_2_b0x1_addr_1, %op_2_b0x1_col
  %op_2_b0x1_byte_1 = mul i32 %op_2_b0x1_addr2_1, 2
  %op_2_b0x1_byte64_1 = zext i32 %op_2_b0x1_byte_1 to i64
  %op_2_b0x1_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x1_byte64_1
  %op_2_b0x1_typed_1 = bitcast i8 addrspace(3)* %op_2_b0x1_ptr_1 to half addrspace(3)*
  %op_2_b0x1_load_1 = load half, half addrspace(3)* %op_2_b0x1_typed_1
  %op_2_b0x1_v2_a = insertelement <2 x half> undef, half %op_2_b0x1_load_0, i32 0
  %op_2_b0x1_v2 = insertelement <2 x half> %op_2_b0x1_v2_a, half %op_2_b0x1_load_1, i32 1
  %op_2_b0x1_sram_e0 = extractelement <2 x half> %op_2_b0x1_v2, i32 0
  %op_2_b0x1_sram_e1 = extractelement <2 x half> %op_2_b0x1_v2, i32 1
  %op_2_b0x1_sram_v0 = insertelement <64 x half> undef, half %op_2_b0x1_sram_e0, i32 0
  %op_2_b0x1_sram = insertelement <64 x half> %op_2_b0x1_sram_v0, half %op_2_b0x1_sram_e1, i32 1
  %op_2_c0x1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a0_sram, <64 x half> %op_2_b0x1_sram, <64 x float> %op_1_c1x1) #3
  %op_2_b0x2_row = add i32 %morton_x, 16
  %op_2_b0x2_col = add i32 %morton_y, 0
  %op_2_b0x2_r_0 = add i32 %op_2_b0x2_row, 0
  %op_2_b0x2_addr_0 = mul i32 %op_2_b0x2_r_0, 16
  %op_2_b0x2_addr2_0 = add i32 %op_2_b0x2_addr_0, %op_2_b0x2_col
  %op_2_b0x2_byte_0 = mul i32 %op_2_b0x2_addr2_0, 2
  %op_2_b0x2_byte64_0 = zext i32 %op_2_b0x2_byte_0 to i64
  %op_2_b0x2_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x2_byte64_0
  %op_2_b0x2_typed_0 = bitcast i8 addrspace(3)* %op_2_b0x2_ptr_0 to half addrspace(3)*
  %op_2_b0x2_load_0 = load half, half addrspace(3)* %op_2_b0x2_typed_0
  %op_2_b0x2_r_1 = add i32 %op_2_b0x2_row, 1
  %op_2_b0x2_addr_1 = mul i32 %op_2_b0x2_r_1, 16
  %op_2_b0x2_addr2_1 = add i32 %op_2_b0x2_addr_1, %op_2_b0x2_col
  %op_2_b0x2_byte_1 = mul i32 %op_2_b0x2_addr2_1, 2
  %op_2_b0x2_byte64_1 = zext i32 %op_2_b0x2_byte_1 to i64
  %op_2_b0x2_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x2_byte64_1
  %op_2_b0x2_typed_1 = bitcast i8 addrspace(3)* %op_2_b0x2_ptr_1 to half addrspace(3)*
  %op_2_b0x2_load_1 = load half, half addrspace(3)* %op_2_b0x2_typed_1
  %op_2_b0x2_v2_a = insertelement <2 x half> undef, half %op_2_b0x2_load_0, i32 0
  %op_2_b0x2_v2 = insertelement <2 x half> %op_2_b0x2_v2_a, half %op_2_b0x2_load_1, i32 1
  %op_2_b0x2_sram_e0 = extractelement <2 x half> %op_2_b0x2_v2, i32 0
  %op_2_b0x2_sram_e1 = extractelement <2 x half> %op_2_b0x2_v2, i32 1
  %op_2_b0x2_sram_v0 = insertelement <64 x half> undef, half %op_2_b0x2_sram_e0, i32 0
  %op_2_b0x2_sram = insertelement <64 x half> %op_2_b0x2_sram_v0, half %op_2_b0x2_sram_e1, i32 1
  %op_2_c0x2 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a0_sram, <64 x half> %op_2_b0x2_sram, <64 x float> %op_1_c1x2) #3
  %op_2_b0x3_row = add i32 %morton_x, 24
  %op_2_b0x3_col = add i32 %morton_y, 0
  %op_2_b0x3_r_0 = add i32 %op_2_b0x3_row, 0
  %op_2_b0x3_addr_0 = mul i32 %op_2_b0x3_r_0, 16
  %op_2_b0x3_addr2_0 = add i32 %op_2_b0x3_addr_0, %op_2_b0x3_col
  %op_2_b0x3_byte_0 = mul i32 %op_2_b0x3_addr2_0, 2
  %op_2_b0x3_byte64_0 = zext i32 %op_2_b0x3_byte_0 to i64
  %op_2_b0x3_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x3_byte64_0
  %op_2_b0x3_typed_0 = bitcast i8 addrspace(3)* %op_2_b0x3_ptr_0 to half addrspace(3)*
  %op_2_b0x3_load_0 = load half, half addrspace(3)* %op_2_b0x3_typed_0
  %op_2_b0x3_r_1 = add i32 %op_2_b0x3_row, 1
  %op_2_b0x3_addr_1 = mul i32 %op_2_b0x3_r_1, 16
  %op_2_b0x3_addr2_1 = add i32 %op_2_b0x3_addr_1, %op_2_b0x3_col
  %op_2_b0x3_byte_1 = mul i32 %op_2_b0x3_addr2_1, 2
  %op_2_b0x3_byte64_1 = zext i32 %op_2_b0x3_byte_1 to i64
  %op_2_b0x3_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x3_byte64_1
  %op_2_b0x3_typed_1 = bitcast i8 addrspace(3)* %op_2_b0x3_ptr_1 to half addrspace(3)*
  %op_2_b0x3_load_1 = load half, half addrspace(3)* %op_2_b0x3_typed_1
  %op_2_b0x3_v2_a = insertelement <2 x half> undef, half %op_2_b0x3_load_0, i32 0
  %op_2_b0x3_v2 = insertelement <2 x half> %op_2_b0x3_v2_a, half %op_2_b0x3_load_1, i32 1
  %op_2_b0x3_sram_e0 = extractelement <2 x half> %op_2_b0x3_v2, i32 0
  %op_2_b0x3_sram_e1 = extractelement <2 x half> %op_2_b0x3_v2, i32 1
  %op_2_b0x3_sram_v0 = insertelement <64 x half> undef, half %op_2_b0x3_sram_e0, i32 0
  %op_2_b0x3_sram = insertelement <64 x half> %op_2_b0x3_sram_v0, half %op_2_b0x3_sram_e1, i32 1
  %op_2_c0x3 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a0_sram, <64 x half> %op_2_b0x3_sram, <64 x float> %op_1_c1x3) #3
  %op_2_b0x4_row = add i32 %morton_x, 32
  %op_2_b0x4_col = add i32 %morton_y, 0
  %op_2_b0x4_r_0 = add i32 %op_2_b0x4_row, 0
  %op_2_b0x4_addr_0 = mul i32 %op_2_b0x4_r_0, 16
  %op_2_b0x4_addr2_0 = add i32 %op_2_b0x4_addr_0, %op_2_b0x4_col
  %op_2_b0x4_byte_0 = mul i32 %op_2_b0x4_addr2_0, 2
  %op_2_b0x4_byte64_0 = zext i32 %op_2_b0x4_byte_0 to i64
  %op_2_b0x4_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x4_byte64_0
  %op_2_b0x4_typed_0 = bitcast i8 addrspace(3)* %op_2_b0x4_ptr_0 to half addrspace(3)*
  %op_2_b0x4_load_0 = load half, half addrspace(3)* %op_2_b0x4_typed_0
  %op_2_b0x4_r_1 = add i32 %op_2_b0x4_row, 1
  %op_2_b0x4_addr_1 = mul i32 %op_2_b0x4_r_1, 16
  %op_2_b0x4_addr2_1 = add i32 %op_2_b0x4_addr_1, %op_2_b0x4_col
  %op_2_b0x4_byte_1 = mul i32 %op_2_b0x4_addr2_1, 2
  %op_2_b0x4_byte64_1 = zext i32 %op_2_b0x4_byte_1 to i64
  %op_2_b0x4_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x4_byte64_1
  %op_2_b0x4_typed_1 = bitcast i8 addrspace(3)* %op_2_b0x4_ptr_1 to half addrspace(3)*
  %op_2_b0x4_load_1 = load half, half addrspace(3)* %op_2_b0x4_typed_1
  %op_2_b0x4_v2_a = insertelement <2 x half> undef, half %op_2_b0x4_load_0, i32 0
  %op_2_b0x4_v2 = insertelement <2 x half> %op_2_b0x4_v2_a, half %op_2_b0x4_load_1, i32 1
  %op_2_b0x4_sram_e0 = extractelement <2 x half> %op_2_b0x4_v2, i32 0
  %op_2_b0x4_sram_e1 = extractelement <2 x half> %op_2_b0x4_v2, i32 1
  %op_2_b0x4_sram_v0 = insertelement <64 x half> undef, half %op_2_b0x4_sram_e0, i32 0
  %op_2_b0x4_sram = insertelement <64 x half> %op_2_b0x4_sram_v0, half %op_2_b0x4_sram_e1, i32 1
  %op_2_c0x4 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a0_sram, <64 x half> %op_2_b0x4_sram, <64 x float> %op_1_c1x4) #3
  %op_2_b0x5_row = add i32 %morton_x, 40
  %op_2_b0x5_col = add i32 %morton_y, 0
  %op_2_b0x5_r_0 = add i32 %op_2_b0x5_row, 0
  %op_2_b0x5_addr_0 = mul i32 %op_2_b0x5_r_0, 16
  %op_2_b0x5_addr2_0 = add i32 %op_2_b0x5_addr_0, %op_2_b0x5_col
  %op_2_b0x5_byte_0 = mul i32 %op_2_b0x5_addr2_0, 2
  %op_2_b0x5_byte64_0 = zext i32 %op_2_b0x5_byte_0 to i64
  %op_2_b0x5_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x5_byte64_0
  %op_2_b0x5_typed_0 = bitcast i8 addrspace(3)* %op_2_b0x5_ptr_0 to half addrspace(3)*
  %op_2_b0x5_load_0 = load half, half addrspace(3)* %op_2_b0x5_typed_0
  %op_2_b0x5_r_1 = add i32 %op_2_b0x5_row, 1
  %op_2_b0x5_addr_1 = mul i32 %op_2_b0x5_r_1, 16
  %op_2_b0x5_addr2_1 = add i32 %op_2_b0x5_addr_1, %op_2_b0x5_col
  %op_2_b0x5_byte_1 = mul i32 %op_2_b0x5_addr2_1, 2
  %op_2_b0x5_byte64_1 = zext i32 %op_2_b0x5_byte_1 to i64
  %op_2_b0x5_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x5_byte64_1
  %op_2_b0x5_typed_1 = bitcast i8 addrspace(3)* %op_2_b0x5_ptr_1 to half addrspace(3)*
  %op_2_b0x5_load_1 = load half, half addrspace(3)* %op_2_b0x5_typed_1
  %op_2_b0x5_v2_a = insertelement <2 x half> undef, half %op_2_b0x5_load_0, i32 0
  %op_2_b0x5_v2 = insertelement <2 x half> %op_2_b0x5_v2_a, half %op_2_b0x5_load_1, i32 1
  %op_2_b0x5_sram_e0 = extractelement <2 x half> %op_2_b0x5_v2, i32 0
  %op_2_b0x5_sram_e1 = extractelement <2 x half> %op_2_b0x5_v2, i32 1
  %op_2_b0x5_sram_v0 = insertelement <64 x half> undef, half %op_2_b0x5_sram_e0, i32 0
  %op_2_b0x5_sram = insertelement <64 x half> %op_2_b0x5_sram_v0, half %op_2_b0x5_sram_e1, i32 1
  %op_2_c0x5 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a0_sram, <64 x half> %op_2_b0x5_sram, <64 x float> %op_1_c1x5) #3
  %op_2_b0x6_row = add i32 %morton_x, 48
  %op_2_b0x6_col = add i32 %morton_y, 0
  %op_2_b0x6_r_0 = add i32 %op_2_b0x6_row, 0
  %op_2_b0x6_addr_0 = mul i32 %op_2_b0x6_r_0, 16
  %op_2_b0x6_addr2_0 = add i32 %op_2_b0x6_addr_0, %op_2_b0x6_col
  %op_2_b0x6_byte_0 = mul i32 %op_2_b0x6_addr2_0, 2
  %op_2_b0x6_byte64_0 = zext i32 %op_2_b0x6_byte_0 to i64
  %op_2_b0x6_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x6_byte64_0
  %op_2_b0x6_typed_0 = bitcast i8 addrspace(3)* %op_2_b0x6_ptr_0 to half addrspace(3)*
  %op_2_b0x6_load_0 = load half, half addrspace(3)* %op_2_b0x6_typed_0
  %op_2_b0x6_r_1 = add i32 %op_2_b0x6_row, 1
  %op_2_b0x6_addr_1 = mul i32 %op_2_b0x6_r_1, 16
  %op_2_b0x6_addr2_1 = add i32 %op_2_b0x6_addr_1, %op_2_b0x6_col
  %op_2_b0x6_byte_1 = mul i32 %op_2_b0x6_addr2_1, 2
  %op_2_b0x6_byte64_1 = zext i32 %op_2_b0x6_byte_1 to i64
  %op_2_b0x6_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x6_byte64_1
  %op_2_b0x6_typed_1 = bitcast i8 addrspace(3)* %op_2_b0x6_ptr_1 to half addrspace(3)*
  %op_2_b0x6_load_1 = load half, half addrspace(3)* %op_2_b0x6_typed_1
  %op_2_b0x6_v2_a = insertelement <2 x half> undef, half %op_2_b0x6_load_0, i32 0
  %op_2_b0x6_v2 = insertelement <2 x half> %op_2_b0x6_v2_a, half %op_2_b0x6_load_1, i32 1
  %op_2_b0x6_sram_e0 = extractelement <2 x half> %op_2_b0x6_v2, i32 0
  %op_2_b0x6_sram_e1 = extractelement <2 x half> %op_2_b0x6_v2, i32 1
  %op_2_b0x6_sram_v0 = insertelement <64 x half> undef, half %op_2_b0x6_sram_e0, i32 0
  %op_2_b0x6_sram = insertelement <64 x half> %op_2_b0x6_sram_v0, half %op_2_b0x6_sram_e1, i32 1
  %op_2_c0x6 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a0_sram, <64 x half> %op_2_b0x6_sram, <64 x float> %op_1_c1x6) #3
  %op_2_b0x7_row = add i32 %morton_x, 56
  %op_2_b0x7_col = add i32 %morton_y, 0
  %op_2_b0x7_r_0 = add i32 %op_2_b0x7_row, 0
  %op_2_b0x7_addr_0 = mul i32 %op_2_b0x7_r_0, 16
  %op_2_b0x7_addr2_0 = add i32 %op_2_b0x7_addr_0, %op_2_b0x7_col
  %op_2_b0x7_byte_0 = mul i32 %op_2_b0x7_addr2_0, 2
  %op_2_b0x7_byte64_0 = zext i32 %op_2_b0x7_byte_0 to i64
  %op_2_b0x7_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x7_byte64_0
  %op_2_b0x7_typed_0 = bitcast i8 addrspace(3)* %op_2_b0x7_ptr_0 to half addrspace(3)*
  %op_2_b0x7_load_0 = load half, half addrspace(3)* %op_2_b0x7_typed_0
  %op_2_b0x7_r_1 = add i32 %op_2_b0x7_row, 1
  %op_2_b0x7_addr_1 = mul i32 %op_2_b0x7_r_1, 16
  %op_2_b0x7_addr2_1 = add i32 %op_2_b0x7_addr_1, %op_2_b0x7_col
  %op_2_b0x7_byte_1 = mul i32 %op_2_b0x7_addr2_1, 2
  %op_2_b0x7_byte64_1 = zext i32 %op_2_b0x7_byte_1 to i64
  %op_2_b0x7_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b0x7_byte64_1
  %op_2_b0x7_typed_1 = bitcast i8 addrspace(3)* %op_2_b0x7_ptr_1 to half addrspace(3)*
  %op_2_b0x7_load_1 = load half, half addrspace(3)* %op_2_b0x7_typed_1
  %op_2_b0x7_v2_a = insertelement <2 x half> undef, half %op_2_b0x7_load_0, i32 0
  %op_2_b0x7_v2 = insertelement <2 x half> %op_2_b0x7_v2_a, half %op_2_b0x7_load_1, i32 1
  %op_2_b0x7_sram_e0 = extractelement <2 x half> %op_2_b0x7_v2, i32 0
  %op_2_b0x7_sram_e1 = extractelement <2 x half> %op_2_b0x7_v2, i32 1
  %op_2_b0x7_sram_v0 = insertelement <64 x half> undef, half %op_2_b0x7_sram_e0, i32 0
  %op_2_b0x7_sram = insertelement <64 x half> %op_2_b0x7_sram_v0, half %op_2_b0x7_sram_e1, i32 1
  %op_2_c0x7 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a0_sram, <64 x half> %op_2_b0x7_sram, <64 x float> %op_1_c1x7) #3
  %op_2_a1_seq = add i32 %clamped_par_off, 0
  %op_2_a1_head = add i32 %morton_x, 40
  %op_2_a1_addr = mul i32 %op_2_a1_seq, 64
  %op_2_a1_addr2 = add i32 %op_2_a1_addr, %op_2_a1_head
  %op_2_a1_byte = mul i32 %op_2_a1_addr2, 2
  %op_2_a1_byte64 = zext i32 %op_2_a1_byte to i64
  %op_2_a1_ptr = getelementptr i8, i8 addrspace(1)* %Q, i64 %op_2_a1_byte64
  %op_2_a1_typed = bitcast i8 addrspace(1)* %op_2_a1_ptr to <2 x half> addrspace(1)*
  %op_2_a1_load = load <2 x half>, <2 x half> addrspace(1)* %op_2_a1_typed, align 4
  %op_2_a1_v2 = bitcast <2 x half> %op_2_a1_load to <2 x half>
  %op_2_a1_sram_e0 = extractelement <2 x half> %op_2_a1_v2, i32 0
  %op_2_a1_sram_e1 = extractelement <2 x half> %op_2_a1_v2, i32 1
  %op_2_a1_sram_v0 = insertelement <64 x half> undef, half %op_2_a1_sram_e0, i32 0
  %op_2_a1_sram = insertelement <64 x half> %op_2_a1_sram_v0, half %op_2_a1_sram_e1, i32 1
  %op_2_b1x0_row = add i32 %morton_x, 0
  %op_2_b1x0_col = add i32 %morton_y, 8
  %op_2_b1x0_r_0 = add i32 %op_2_b1x0_row, 0
  %op_2_b1x0_addr_0 = mul i32 %op_2_b1x0_r_0, 16
  %op_2_b1x0_addr2_0 = add i32 %op_2_b1x0_addr_0, %op_2_b1x0_col
  %op_2_b1x0_byte_0 = mul i32 %op_2_b1x0_addr2_0, 2
  %op_2_b1x0_byte64_0 = zext i32 %op_2_b1x0_byte_0 to i64
  %op_2_b1x0_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x0_byte64_0
  %op_2_b1x0_typed_0 = bitcast i8 addrspace(3)* %op_2_b1x0_ptr_0 to half addrspace(3)*
  %op_2_b1x0_load_0 = load half, half addrspace(3)* %op_2_b1x0_typed_0
  %op_2_b1x0_r_1 = add i32 %op_2_b1x0_row, 1
  %op_2_b1x0_addr_1 = mul i32 %op_2_b1x0_r_1, 16
  %op_2_b1x0_addr2_1 = add i32 %op_2_b1x0_addr_1, %op_2_b1x0_col
  %op_2_b1x0_byte_1 = mul i32 %op_2_b1x0_addr2_1, 2
  %op_2_b1x0_byte64_1 = zext i32 %op_2_b1x0_byte_1 to i64
  %op_2_b1x0_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x0_byte64_1
  %op_2_b1x0_typed_1 = bitcast i8 addrspace(3)* %op_2_b1x0_ptr_1 to half addrspace(3)*
  %op_2_b1x0_load_1 = load half, half addrspace(3)* %op_2_b1x0_typed_1
  %op_2_b1x0_v2_a = insertelement <2 x half> undef, half %op_2_b1x0_load_0, i32 0
  %op_2_b1x0_v2 = insertelement <2 x half> %op_2_b1x0_v2_a, half %op_2_b1x0_load_1, i32 1
  %op_2_b1x0_sram_e0 = extractelement <2 x half> %op_2_b1x0_v2, i32 0
  %op_2_b1x0_sram_e1 = extractelement <2 x half> %op_2_b1x0_v2, i32 1
  %op_2_b1x0_sram_v0 = insertelement <64 x half> undef, half %op_2_b1x0_sram_e0, i32 0
  %op_2_b1x0_sram = insertelement <64 x half> %op_2_b1x0_sram_v0, half %op_2_b1x0_sram_e1, i32 1
  %op_2_c1x0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a1_sram, <64 x half> %op_2_b1x0_sram, <64 x float> %op_2_c0x0) #3
  %op_2_b1x1_row = add i32 %morton_x, 8
  %op_2_b1x1_col = add i32 %morton_y, 8
  %op_2_b1x1_r_0 = add i32 %op_2_b1x1_row, 0
  %op_2_b1x1_addr_0 = mul i32 %op_2_b1x1_r_0, 16
  %op_2_b1x1_addr2_0 = add i32 %op_2_b1x1_addr_0, %op_2_b1x1_col
  %op_2_b1x1_byte_0 = mul i32 %op_2_b1x1_addr2_0, 2
  %op_2_b1x1_byte64_0 = zext i32 %op_2_b1x1_byte_0 to i64
  %op_2_b1x1_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x1_byte64_0
  %op_2_b1x1_typed_0 = bitcast i8 addrspace(3)* %op_2_b1x1_ptr_0 to half addrspace(3)*
  %op_2_b1x1_load_0 = load half, half addrspace(3)* %op_2_b1x1_typed_0
  %op_2_b1x1_r_1 = add i32 %op_2_b1x1_row, 1
  %op_2_b1x1_addr_1 = mul i32 %op_2_b1x1_r_1, 16
  %op_2_b1x1_addr2_1 = add i32 %op_2_b1x1_addr_1, %op_2_b1x1_col
  %op_2_b1x1_byte_1 = mul i32 %op_2_b1x1_addr2_1, 2
  %op_2_b1x1_byte64_1 = zext i32 %op_2_b1x1_byte_1 to i64
  %op_2_b1x1_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x1_byte64_1
  %op_2_b1x1_typed_1 = bitcast i8 addrspace(3)* %op_2_b1x1_ptr_1 to half addrspace(3)*
  %op_2_b1x1_load_1 = load half, half addrspace(3)* %op_2_b1x1_typed_1
  %op_2_b1x1_v2_a = insertelement <2 x half> undef, half %op_2_b1x1_load_0, i32 0
  %op_2_b1x1_v2 = insertelement <2 x half> %op_2_b1x1_v2_a, half %op_2_b1x1_load_1, i32 1
  %op_2_b1x1_sram_e0 = extractelement <2 x half> %op_2_b1x1_v2, i32 0
  %op_2_b1x1_sram_e1 = extractelement <2 x half> %op_2_b1x1_v2, i32 1
  %op_2_b1x1_sram_v0 = insertelement <64 x half> undef, half %op_2_b1x1_sram_e0, i32 0
  %op_2_b1x1_sram = insertelement <64 x half> %op_2_b1x1_sram_v0, half %op_2_b1x1_sram_e1, i32 1
  %op_2_c1x1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a1_sram, <64 x half> %op_2_b1x1_sram, <64 x float> %op_2_c0x1) #3
  %op_2_b1x2_row = add i32 %morton_x, 16
  %op_2_b1x2_col = add i32 %morton_y, 8
  %op_2_b1x2_r_0 = add i32 %op_2_b1x2_row, 0
  %op_2_b1x2_addr_0 = mul i32 %op_2_b1x2_r_0, 16
  %op_2_b1x2_addr2_0 = add i32 %op_2_b1x2_addr_0, %op_2_b1x2_col
  %op_2_b1x2_byte_0 = mul i32 %op_2_b1x2_addr2_0, 2
  %op_2_b1x2_byte64_0 = zext i32 %op_2_b1x2_byte_0 to i64
  %op_2_b1x2_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x2_byte64_0
  %op_2_b1x2_typed_0 = bitcast i8 addrspace(3)* %op_2_b1x2_ptr_0 to half addrspace(3)*
  %op_2_b1x2_load_0 = load half, half addrspace(3)* %op_2_b1x2_typed_0
  %op_2_b1x2_r_1 = add i32 %op_2_b1x2_row, 1
  %op_2_b1x2_addr_1 = mul i32 %op_2_b1x2_r_1, 16
  %op_2_b1x2_addr2_1 = add i32 %op_2_b1x2_addr_1, %op_2_b1x2_col
  %op_2_b1x2_byte_1 = mul i32 %op_2_b1x2_addr2_1, 2
  %op_2_b1x2_byte64_1 = zext i32 %op_2_b1x2_byte_1 to i64
  %op_2_b1x2_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x2_byte64_1
  %op_2_b1x2_typed_1 = bitcast i8 addrspace(3)* %op_2_b1x2_ptr_1 to half addrspace(3)*
  %op_2_b1x2_load_1 = load half, half addrspace(3)* %op_2_b1x2_typed_1
  %op_2_b1x2_v2_a = insertelement <2 x half> undef, half %op_2_b1x2_load_0, i32 0
  %op_2_b1x2_v2 = insertelement <2 x half> %op_2_b1x2_v2_a, half %op_2_b1x2_load_1, i32 1
  %op_2_b1x2_sram_e0 = extractelement <2 x half> %op_2_b1x2_v2, i32 0
  %op_2_b1x2_sram_e1 = extractelement <2 x half> %op_2_b1x2_v2, i32 1
  %op_2_b1x2_sram_v0 = insertelement <64 x half> undef, half %op_2_b1x2_sram_e0, i32 0
  %op_2_b1x2_sram = insertelement <64 x half> %op_2_b1x2_sram_v0, half %op_2_b1x2_sram_e1, i32 1
  %op_2_c1x2 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a1_sram, <64 x half> %op_2_b1x2_sram, <64 x float> %op_2_c0x2) #3
  %op_2_b1x3_row = add i32 %morton_x, 24
  %op_2_b1x3_col = add i32 %morton_y, 8
  %op_2_b1x3_r_0 = add i32 %op_2_b1x3_row, 0
  %op_2_b1x3_addr_0 = mul i32 %op_2_b1x3_r_0, 16
  %op_2_b1x3_addr2_0 = add i32 %op_2_b1x3_addr_0, %op_2_b1x3_col
  %op_2_b1x3_byte_0 = mul i32 %op_2_b1x3_addr2_0, 2
  %op_2_b1x3_byte64_0 = zext i32 %op_2_b1x3_byte_0 to i64
  %op_2_b1x3_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x3_byte64_0
  %op_2_b1x3_typed_0 = bitcast i8 addrspace(3)* %op_2_b1x3_ptr_0 to half addrspace(3)*
  %op_2_b1x3_load_0 = load half, half addrspace(3)* %op_2_b1x3_typed_0
  %op_2_b1x3_r_1 = add i32 %op_2_b1x3_row, 1
  %op_2_b1x3_addr_1 = mul i32 %op_2_b1x3_r_1, 16
  %op_2_b1x3_addr2_1 = add i32 %op_2_b1x3_addr_1, %op_2_b1x3_col
  %op_2_b1x3_byte_1 = mul i32 %op_2_b1x3_addr2_1, 2
  %op_2_b1x3_byte64_1 = zext i32 %op_2_b1x3_byte_1 to i64
  %op_2_b1x3_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x3_byte64_1
  %op_2_b1x3_typed_1 = bitcast i8 addrspace(3)* %op_2_b1x3_ptr_1 to half addrspace(3)*
  %op_2_b1x3_load_1 = load half, half addrspace(3)* %op_2_b1x3_typed_1
  %op_2_b1x3_v2_a = insertelement <2 x half> undef, half %op_2_b1x3_load_0, i32 0
  %op_2_b1x3_v2 = insertelement <2 x half> %op_2_b1x3_v2_a, half %op_2_b1x3_load_1, i32 1
  %op_2_b1x3_sram_e0 = extractelement <2 x half> %op_2_b1x3_v2, i32 0
  %op_2_b1x3_sram_e1 = extractelement <2 x half> %op_2_b1x3_v2, i32 1
  %op_2_b1x3_sram_v0 = insertelement <64 x half> undef, half %op_2_b1x3_sram_e0, i32 0
  %op_2_b1x3_sram = insertelement <64 x half> %op_2_b1x3_sram_v0, half %op_2_b1x3_sram_e1, i32 1
  %op_2_c1x3 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a1_sram, <64 x half> %op_2_b1x3_sram, <64 x float> %op_2_c0x3) #3
  %op_2_b1x4_row = add i32 %morton_x, 32
  %op_2_b1x4_col = add i32 %morton_y, 8
  %op_2_b1x4_r_0 = add i32 %op_2_b1x4_row, 0
  %op_2_b1x4_addr_0 = mul i32 %op_2_b1x4_r_0, 16
  %op_2_b1x4_addr2_0 = add i32 %op_2_b1x4_addr_0, %op_2_b1x4_col
  %op_2_b1x4_byte_0 = mul i32 %op_2_b1x4_addr2_0, 2
  %op_2_b1x4_byte64_0 = zext i32 %op_2_b1x4_byte_0 to i64
  %op_2_b1x4_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x4_byte64_0
  %op_2_b1x4_typed_0 = bitcast i8 addrspace(3)* %op_2_b1x4_ptr_0 to half addrspace(3)*
  %op_2_b1x4_load_0 = load half, half addrspace(3)* %op_2_b1x4_typed_0
  %op_2_b1x4_r_1 = add i32 %op_2_b1x4_row, 1
  %op_2_b1x4_addr_1 = mul i32 %op_2_b1x4_r_1, 16
  %op_2_b1x4_addr2_1 = add i32 %op_2_b1x4_addr_1, %op_2_b1x4_col
  %op_2_b1x4_byte_1 = mul i32 %op_2_b1x4_addr2_1, 2
  %op_2_b1x4_byte64_1 = zext i32 %op_2_b1x4_byte_1 to i64
  %op_2_b1x4_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x4_byte64_1
  %op_2_b1x4_typed_1 = bitcast i8 addrspace(3)* %op_2_b1x4_ptr_1 to half addrspace(3)*
  %op_2_b1x4_load_1 = load half, half addrspace(3)* %op_2_b1x4_typed_1
  %op_2_b1x4_v2_a = insertelement <2 x half> undef, half %op_2_b1x4_load_0, i32 0
  %op_2_b1x4_v2 = insertelement <2 x half> %op_2_b1x4_v2_a, half %op_2_b1x4_load_1, i32 1
  %op_2_b1x4_sram_e0 = extractelement <2 x half> %op_2_b1x4_v2, i32 0
  %op_2_b1x4_sram_e1 = extractelement <2 x half> %op_2_b1x4_v2, i32 1
  %op_2_b1x4_sram_v0 = insertelement <64 x half> undef, half %op_2_b1x4_sram_e0, i32 0
  %op_2_b1x4_sram = insertelement <64 x half> %op_2_b1x4_sram_v0, half %op_2_b1x4_sram_e1, i32 1
  %op_2_c1x4 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a1_sram, <64 x half> %op_2_b1x4_sram, <64 x float> %op_2_c0x4) #3
  %op_2_b1x5_row = add i32 %morton_x, 40
  %op_2_b1x5_col = add i32 %morton_y, 8
  %op_2_b1x5_r_0 = add i32 %op_2_b1x5_row, 0
  %op_2_b1x5_addr_0 = mul i32 %op_2_b1x5_r_0, 16
  %op_2_b1x5_addr2_0 = add i32 %op_2_b1x5_addr_0, %op_2_b1x5_col
  %op_2_b1x5_byte_0 = mul i32 %op_2_b1x5_addr2_0, 2
  %op_2_b1x5_byte64_0 = zext i32 %op_2_b1x5_byte_0 to i64
  %op_2_b1x5_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x5_byte64_0
  %op_2_b1x5_typed_0 = bitcast i8 addrspace(3)* %op_2_b1x5_ptr_0 to half addrspace(3)*
  %op_2_b1x5_load_0 = load half, half addrspace(3)* %op_2_b1x5_typed_0
  %op_2_b1x5_r_1 = add i32 %op_2_b1x5_row, 1
  %op_2_b1x5_addr_1 = mul i32 %op_2_b1x5_r_1, 16
  %op_2_b1x5_addr2_1 = add i32 %op_2_b1x5_addr_1, %op_2_b1x5_col
  %op_2_b1x5_byte_1 = mul i32 %op_2_b1x5_addr2_1, 2
  %op_2_b1x5_byte64_1 = zext i32 %op_2_b1x5_byte_1 to i64
  %op_2_b1x5_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x5_byte64_1
  %op_2_b1x5_typed_1 = bitcast i8 addrspace(3)* %op_2_b1x5_ptr_1 to half addrspace(3)*
  %op_2_b1x5_load_1 = load half, half addrspace(3)* %op_2_b1x5_typed_1
  %op_2_b1x5_v2_a = insertelement <2 x half> undef, half %op_2_b1x5_load_0, i32 0
  %op_2_b1x5_v2 = insertelement <2 x half> %op_2_b1x5_v2_a, half %op_2_b1x5_load_1, i32 1
  %op_2_b1x5_sram_e0 = extractelement <2 x half> %op_2_b1x5_v2, i32 0
  %op_2_b1x5_sram_e1 = extractelement <2 x half> %op_2_b1x5_v2, i32 1
  %op_2_b1x5_sram_v0 = insertelement <64 x half> undef, half %op_2_b1x5_sram_e0, i32 0
  %op_2_b1x5_sram = insertelement <64 x half> %op_2_b1x5_sram_v0, half %op_2_b1x5_sram_e1, i32 1
  %op_2_c1x5 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a1_sram, <64 x half> %op_2_b1x5_sram, <64 x float> %op_2_c0x5) #3
  %op_2_b1x6_row = add i32 %morton_x, 48
  %op_2_b1x6_col = add i32 %morton_y, 8
  %op_2_b1x6_r_0 = add i32 %op_2_b1x6_row, 0
  %op_2_b1x6_addr_0 = mul i32 %op_2_b1x6_r_0, 16
  %op_2_b1x6_addr2_0 = add i32 %op_2_b1x6_addr_0, %op_2_b1x6_col
  %op_2_b1x6_byte_0 = mul i32 %op_2_b1x6_addr2_0, 2
  %op_2_b1x6_byte64_0 = zext i32 %op_2_b1x6_byte_0 to i64
  %op_2_b1x6_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x6_byte64_0
  %op_2_b1x6_typed_0 = bitcast i8 addrspace(3)* %op_2_b1x6_ptr_0 to half addrspace(3)*
  %op_2_b1x6_load_0 = load half, half addrspace(3)* %op_2_b1x6_typed_0
  %op_2_b1x6_r_1 = add i32 %op_2_b1x6_row, 1
  %op_2_b1x6_addr_1 = mul i32 %op_2_b1x6_r_1, 16
  %op_2_b1x6_addr2_1 = add i32 %op_2_b1x6_addr_1, %op_2_b1x6_col
  %op_2_b1x6_byte_1 = mul i32 %op_2_b1x6_addr2_1, 2
  %op_2_b1x6_byte64_1 = zext i32 %op_2_b1x6_byte_1 to i64
  %op_2_b1x6_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x6_byte64_1
  %op_2_b1x6_typed_1 = bitcast i8 addrspace(3)* %op_2_b1x6_ptr_1 to half addrspace(3)*
  %op_2_b1x6_load_1 = load half, half addrspace(3)* %op_2_b1x6_typed_1
  %op_2_b1x6_v2_a = insertelement <2 x half> undef, half %op_2_b1x6_load_0, i32 0
  %op_2_b1x6_v2 = insertelement <2 x half> %op_2_b1x6_v2_a, half %op_2_b1x6_load_1, i32 1
  %op_2_b1x6_sram_e0 = extractelement <2 x half> %op_2_b1x6_v2, i32 0
  %op_2_b1x6_sram_e1 = extractelement <2 x half> %op_2_b1x6_v2, i32 1
  %op_2_b1x6_sram_v0 = insertelement <64 x half> undef, half %op_2_b1x6_sram_e0, i32 0
  %op_2_b1x6_sram = insertelement <64 x half> %op_2_b1x6_sram_v0, half %op_2_b1x6_sram_e1, i32 1
  %op_2_c1x6 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a1_sram, <64 x half> %op_2_b1x6_sram, <64 x float> %op_2_c0x6) #3
  %op_2_b1x7_row = add i32 %morton_x, 56
  %op_2_b1x7_col = add i32 %morton_y, 8
  %op_2_b1x7_r_0 = add i32 %op_2_b1x7_row, 0
  %op_2_b1x7_addr_0 = mul i32 %op_2_b1x7_r_0, 16
  %op_2_b1x7_addr2_0 = add i32 %op_2_b1x7_addr_0, %op_2_b1x7_col
  %op_2_b1x7_byte_0 = mul i32 %op_2_b1x7_addr2_0, 2
  %op_2_b1x7_byte64_0 = zext i32 %op_2_b1x7_byte_0 to i64
  %op_2_b1x7_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x7_byte64_0
  %op_2_b1x7_typed_0 = bitcast i8 addrspace(3)* %op_2_b1x7_ptr_0 to half addrspace(3)*
  %op_2_b1x7_load_0 = load half, half addrspace(3)* %op_2_b1x7_typed_0
  %op_2_b1x7_r_1 = add i32 %op_2_b1x7_row, 1
  %op_2_b1x7_addr_1 = mul i32 %op_2_b1x7_r_1, 16
  %op_2_b1x7_addr2_1 = add i32 %op_2_b1x7_addr_1, %op_2_b1x7_col
  %op_2_b1x7_byte_1 = mul i32 %op_2_b1x7_addr2_1, 2
  %op_2_b1x7_byte64_1 = zext i32 %op_2_b1x7_byte_1 to i64
  %op_2_b1x7_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_2_b1x7_byte64_1
  %op_2_b1x7_typed_1 = bitcast i8 addrspace(3)* %op_2_b1x7_ptr_1 to half addrspace(3)*
  %op_2_b1x7_load_1 = load half, half addrspace(3)* %op_2_b1x7_typed_1
  %op_2_b1x7_v2_a = insertelement <2 x half> undef, half %op_2_b1x7_load_0, i32 0
  %op_2_b1x7_v2 = insertelement <2 x half> %op_2_b1x7_v2_a, half %op_2_b1x7_load_1, i32 1
  %op_2_b1x7_sram_e0 = extractelement <2 x half> %op_2_b1x7_v2, i32 0
  %op_2_b1x7_sram_e1 = extractelement <2 x half> %op_2_b1x7_v2, i32 1
  %op_2_b1x7_sram_v0 = insertelement <64 x half> undef, half %op_2_b1x7_sram_e0, i32 0
  %op_2_b1x7_sram = insertelement <64 x half> %op_2_b1x7_sram_v0, half %op_2_b1x7_sram_e1, i32 1
  %op_2_c1x7 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_2_a1_sram, <64 x half> %op_2_b1x7_sram, <64 x float> %op_2_c0x7) #3
  call void @air.wg.barrier(i32 2, i32 1)

  ; === Sync copy (op_3_b) — all threads cooperative ===
  %op_3_bsc_t0 = mul i32 %sidx, 32
  %op_3_bsc_tid = add i32 %op_3_bsc_t0, %lane_id
  %op_3_bsc_drem = sub i32 64, 48
  %op_3_bsc_dcmp = icmp ult i32 %op_3_bsc_drem, 16
  %op_3_bsc_dsrc = select i1 %op_3_bsc_dcmp, i32 %op_3_bsc_drem, i32 16
  %op_3_bsc_srem = sub i32 64, %c
  %op_3_bsc_scmp = icmp ult i32 %op_3_bsc_srem, 64
  %op_3_bsc_ssrc = select i1 %op_3_bsc_scmp, i32 %op_3_bsc_srem, i32 64
  br label %op_3_bsc_pre

op_3_bsc_pre:
  br label %op_3_bsc_hdr

op_3_bsc_hdr:
  %op_3_bsc_i = phi i32 [%op_3_bsc_tid, %op_3_bsc_pre], [%op_3_bsc_inx, %op_3_bsc_st]
  %op_3_bsc_done = icmp uge i32 %op_3_bsc_i, 1024
  br i1 %op_3_bsc_done, label %op_3_bsc_end, label %op_3_bsc_body

op_3_bsc_body:
  %op_3_bsc_row = lshr i32 %op_3_bsc_i, 4
  %op_3_bsc_col = and i32 %op_3_bsc_i, 15
  %op_3_bsc_rok = icmp ult i32 %op_3_bsc_row, %op_3_bsc_ssrc
  %op_3_bsc_cok = icmp ult i32 %op_3_bsc_col, %op_3_bsc_dsrc
  %op_3_bsc_ib = and i1 %op_3_bsc_rok, %op_3_bsc_cok
  br i1 %op_3_bsc_ib, label %op_3_bsc_ld, label %op_3_bsc_zr

op_3_bsc_ld:
  %op_3_bsc_sr = add i32 %c, %op_3_bsc_row
  %op_3_bsc_sa = mul i32 %op_3_bsc_sr, 64
  %op_3_bsc_sc = add i32 48, %op_3_bsc_col
  %op_3_bsc_sad = add i32 %op_3_bsc_sa, %op_3_bsc_sc
  %op_3_bsc_soff = zext i32 %op_3_bsc_sad to i64
  %op_3_bsc_sbyt = mul i64 %op_3_bsc_soff, 2
  %op_3_bsc_sp = getelementptr i8, i8 addrspace(1)* %K, i64 %op_3_bsc_sbyt
  %op_3_bsc_spt = bitcast i8 addrspace(1)* %op_3_bsc_sp to i16 addrspace(1)*
  %op_3_bsc_lv = load i16, i16 addrspace(1)* %op_3_bsc_spt
  br label %op_3_bsc_st

op_3_bsc_zr:
  br label %op_3_bsc_st

op_3_bsc_st:
  %op_3_bsc_val = phi i16 [%op_3_bsc_lv, %op_3_bsc_ld], [0, %op_3_bsc_zr]
  %op_3_bsc_tr = mul i32 %op_3_bsc_row, 16
  %op_3_bsc_ta = add i32 %op_3_bsc_tr, %op_3_bsc_col
  %op_3_bsc_tb = mul i32 %op_3_bsc_ta, 2
  %op_3_bsc_tb64 = zext i32 %op_3_bsc_tb to i64
  %op_3_bsc_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_bsc_tb64
  %op_3_bsc_tpt = bitcast i8 addrspace(3)* %op_3_bsc_tp to i16 addrspace(3)*
  store i16 %op_3_bsc_val, i16 addrspace(3)* %op_3_bsc_tpt
  %op_3_bsc_inx = add i32 %op_3_bsc_i, 64
  br label %op_3_bsc_hdr

op_3_bsc_end:
  call void @air.wg.barrier(i32 2, i32 1)

  %op_3_a0_seq = add i32 %clamped_par_off, 0
  %op_3_a0_head = add i32 %morton_x, 48
  %op_3_a0_addr = mul i32 %op_3_a0_seq, 64
  %op_3_a0_addr2 = add i32 %op_3_a0_addr, %op_3_a0_head
  %op_3_a0_byte = mul i32 %op_3_a0_addr2, 2
  %op_3_a0_byte64 = zext i32 %op_3_a0_byte to i64
  %op_3_a0_ptr = getelementptr i8, i8 addrspace(1)* %Q, i64 %op_3_a0_byte64
  %op_3_a0_typed = bitcast i8 addrspace(1)* %op_3_a0_ptr to <2 x half> addrspace(1)*
  %op_3_a0_load = load <2 x half>, <2 x half> addrspace(1)* %op_3_a0_typed, align 4
  %op_3_a0_v2 = bitcast <2 x half> %op_3_a0_load to <2 x half>
  %op_3_a0_sram_e0 = extractelement <2 x half> %op_3_a0_v2, i32 0
  %op_3_a0_sram_e1 = extractelement <2 x half> %op_3_a0_v2, i32 1
  %op_3_a0_sram_v0 = insertelement <64 x half> undef, half %op_3_a0_sram_e0, i32 0
  %op_3_a0_sram = insertelement <64 x half> %op_3_a0_sram_v0, half %op_3_a0_sram_e1, i32 1
  %op_3_b0x0_row = add i32 %morton_x, 0
  %op_3_b0x0_col = add i32 %morton_y, 0
  %op_3_b0x0_r_0 = add i32 %op_3_b0x0_row, 0
  %op_3_b0x0_addr_0 = mul i32 %op_3_b0x0_r_0, 16
  %op_3_b0x0_addr2_0 = add i32 %op_3_b0x0_addr_0, %op_3_b0x0_col
  %op_3_b0x0_byte_0 = mul i32 %op_3_b0x0_addr2_0, 2
  %op_3_b0x0_byte64_0 = zext i32 %op_3_b0x0_byte_0 to i64
  %op_3_b0x0_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x0_byte64_0
  %op_3_b0x0_typed_0 = bitcast i8 addrspace(3)* %op_3_b0x0_ptr_0 to half addrspace(3)*
  %op_3_b0x0_load_0 = load half, half addrspace(3)* %op_3_b0x0_typed_0
  %op_3_b0x0_r_1 = add i32 %op_3_b0x0_row, 1
  %op_3_b0x0_addr_1 = mul i32 %op_3_b0x0_r_1, 16
  %op_3_b0x0_addr2_1 = add i32 %op_3_b0x0_addr_1, %op_3_b0x0_col
  %op_3_b0x0_byte_1 = mul i32 %op_3_b0x0_addr2_1, 2
  %op_3_b0x0_byte64_1 = zext i32 %op_3_b0x0_byte_1 to i64
  %op_3_b0x0_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x0_byte64_1
  %op_3_b0x0_typed_1 = bitcast i8 addrspace(3)* %op_3_b0x0_ptr_1 to half addrspace(3)*
  %op_3_b0x0_load_1 = load half, half addrspace(3)* %op_3_b0x0_typed_1
  %op_3_b0x0_v2_a = insertelement <2 x half> undef, half %op_3_b0x0_load_0, i32 0
  %op_3_b0x0_v2 = insertelement <2 x half> %op_3_b0x0_v2_a, half %op_3_b0x0_load_1, i32 1
  %op_3_b0x0_sram_e0 = extractelement <2 x half> %op_3_b0x0_v2, i32 0
  %op_3_b0x0_sram_e1 = extractelement <2 x half> %op_3_b0x0_v2, i32 1
  %op_3_b0x0_sram_v0 = insertelement <64 x half> undef, half %op_3_b0x0_sram_e0, i32 0
  %op_3_b0x0_sram = insertelement <64 x half> %op_3_b0x0_sram_v0, half %op_3_b0x0_sram_e1, i32 1
  %op_3_c0x0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a0_sram, <64 x half> %op_3_b0x0_sram, <64 x float> %op_2_c1x0) #3
  %op_3_b0x1_row = add i32 %morton_x, 8
  %op_3_b0x1_col = add i32 %morton_y, 0
  %op_3_b0x1_r_0 = add i32 %op_3_b0x1_row, 0
  %op_3_b0x1_addr_0 = mul i32 %op_3_b0x1_r_0, 16
  %op_3_b0x1_addr2_0 = add i32 %op_3_b0x1_addr_0, %op_3_b0x1_col
  %op_3_b0x1_byte_0 = mul i32 %op_3_b0x1_addr2_0, 2
  %op_3_b0x1_byte64_0 = zext i32 %op_3_b0x1_byte_0 to i64
  %op_3_b0x1_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x1_byte64_0
  %op_3_b0x1_typed_0 = bitcast i8 addrspace(3)* %op_3_b0x1_ptr_0 to half addrspace(3)*
  %op_3_b0x1_load_0 = load half, half addrspace(3)* %op_3_b0x1_typed_0
  %op_3_b0x1_r_1 = add i32 %op_3_b0x1_row, 1
  %op_3_b0x1_addr_1 = mul i32 %op_3_b0x1_r_1, 16
  %op_3_b0x1_addr2_1 = add i32 %op_3_b0x1_addr_1, %op_3_b0x1_col
  %op_3_b0x1_byte_1 = mul i32 %op_3_b0x1_addr2_1, 2
  %op_3_b0x1_byte64_1 = zext i32 %op_3_b0x1_byte_1 to i64
  %op_3_b0x1_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x1_byte64_1
  %op_3_b0x1_typed_1 = bitcast i8 addrspace(3)* %op_3_b0x1_ptr_1 to half addrspace(3)*
  %op_3_b0x1_load_1 = load half, half addrspace(3)* %op_3_b0x1_typed_1
  %op_3_b0x1_v2_a = insertelement <2 x half> undef, half %op_3_b0x1_load_0, i32 0
  %op_3_b0x1_v2 = insertelement <2 x half> %op_3_b0x1_v2_a, half %op_3_b0x1_load_1, i32 1
  %op_3_b0x1_sram_e0 = extractelement <2 x half> %op_3_b0x1_v2, i32 0
  %op_3_b0x1_sram_e1 = extractelement <2 x half> %op_3_b0x1_v2, i32 1
  %op_3_b0x1_sram_v0 = insertelement <64 x half> undef, half %op_3_b0x1_sram_e0, i32 0
  %op_3_b0x1_sram = insertelement <64 x half> %op_3_b0x1_sram_v0, half %op_3_b0x1_sram_e1, i32 1
  %op_3_c0x1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a0_sram, <64 x half> %op_3_b0x1_sram, <64 x float> %op_2_c1x1) #3
  %op_3_b0x2_row = add i32 %morton_x, 16
  %op_3_b0x2_col = add i32 %morton_y, 0
  %op_3_b0x2_r_0 = add i32 %op_3_b0x2_row, 0
  %op_3_b0x2_addr_0 = mul i32 %op_3_b0x2_r_0, 16
  %op_3_b0x2_addr2_0 = add i32 %op_3_b0x2_addr_0, %op_3_b0x2_col
  %op_3_b0x2_byte_0 = mul i32 %op_3_b0x2_addr2_0, 2
  %op_3_b0x2_byte64_0 = zext i32 %op_3_b0x2_byte_0 to i64
  %op_3_b0x2_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x2_byte64_0
  %op_3_b0x2_typed_0 = bitcast i8 addrspace(3)* %op_3_b0x2_ptr_0 to half addrspace(3)*
  %op_3_b0x2_load_0 = load half, half addrspace(3)* %op_3_b0x2_typed_0
  %op_3_b0x2_r_1 = add i32 %op_3_b0x2_row, 1
  %op_3_b0x2_addr_1 = mul i32 %op_3_b0x2_r_1, 16
  %op_3_b0x2_addr2_1 = add i32 %op_3_b0x2_addr_1, %op_3_b0x2_col
  %op_3_b0x2_byte_1 = mul i32 %op_3_b0x2_addr2_1, 2
  %op_3_b0x2_byte64_1 = zext i32 %op_3_b0x2_byte_1 to i64
  %op_3_b0x2_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x2_byte64_1
  %op_3_b0x2_typed_1 = bitcast i8 addrspace(3)* %op_3_b0x2_ptr_1 to half addrspace(3)*
  %op_3_b0x2_load_1 = load half, half addrspace(3)* %op_3_b0x2_typed_1
  %op_3_b0x2_v2_a = insertelement <2 x half> undef, half %op_3_b0x2_load_0, i32 0
  %op_3_b0x2_v2 = insertelement <2 x half> %op_3_b0x2_v2_a, half %op_3_b0x2_load_1, i32 1
  %op_3_b0x2_sram_e0 = extractelement <2 x half> %op_3_b0x2_v2, i32 0
  %op_3_b0x2_sram_e1 = extractelement <2 x half> %op_3_b0x2_v2, i32 1
  %op_3_b0x2_sram_v0 = insertelement <64 x half> undef, half %op_3_b0x2_sram_e0, i32 0
  %op_3_b0x2_sram = insertelement <64 x half> %op_3_b0x2_sram_v0, half %op_3_b0x2_sram_e1, i32 1
  %op_3_c0x2 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a0_sram, <64 x half> %op_3_b0x2_sram, <64 x float> %op_2_c1x2) #3
  %op_3_b0x3_row = add i32 %morton_x, 24
  %op_3_b0x3_col = add i32 %morton_y, 0
  %op_3_b0x3_r_0 = add i32 %op_3_b0x3_row, 0
  %op_3_b0x3_addr_0 = mul i32 %op_3_b0x3_r_0, 16
  %op_3_b0x3_addr2_0 = add i32 %op_3_b0x3_addr_0, %op_3_b0x3_col
  %op_3_b0x3_byte_0 = mul i32 %op_3_b0x3_addr2_0, 2
  %op_3_b0x3_byte64_0 = zext i32 %op_3_b0x3_byte_0 to i64
  %op_3_b0x3_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x3_byte64_0
  %op_3_b0x3_typed_0 = bitcast i8 addrspace(3)* %op_3_b0x3_ptr_0 to half addrspace(3)*
  %op_3_b0x3_load_0 = load half, half addrspace(3)* %op_3_b0x3_typed_0
  %op_3_b0x3_r_1 = add i32 %op_3_b0x3_row, 1
  %op_3_b0x3_addr_1 = mul i32 %op_3_b0x3_r_1, 16
  %op_3_b0x3_addr2_1 = add i32 %op_3_b0x3_addr_1, %op_3_b0x3_col
  %op_3_b0x3_byte_1 = mul i32 %op_3_b0x3_addr2_1, 2
  %op_3_b0x3_byte64_1 = zext i32 %op_3_b0x3_byte_1 to i64
  %op_3_b0x3_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x3_byte64_1
  %op_3_b0x3_typed_1 = bitcast i8 addrspace(3)* %op_3_b0x3_ptr_1 to half addrspace(3)*
  %op_3_b0x3_load_1 = load half, half addrspace(3)* %op_3_b0x3_typed_1
  %op_3_b0x3_v2_a = insertelement <2 x half> undef, half %op_3_b0x3_load_0, i32 0
  %op_3_b0x3_v2 = insertelement <2 x half> %op_3_b0x3_v2_a, half %op_3_b0x3_load_1, i32 1
  %op_3_b0x3_sram_e0 = extractelement <2 x half> %op_3_b0x3_v2, i32 0
  %op_3_b0x3_sram_e1 = extractelement <2 x half> %op_3_b0x3_v2, i32 1
  %op_3_b0x3_sram_v0 = insertelement <64 x half> undef, half %op_3_b0x3_sram_e0, i32 0
  %op_3_b0x3_sram = insertelement <64 x half> %op_3_b0x3_sram_v0, half %op_3_b0x3_sram_e1, i32 1
  %op_3_c0x3 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a0_sram, <64 x half> %op_3_b0x3_sram, <64 x float> %op_2_c1x3) #3
  %op_3_b0x4_row = add i32 %morton_x, 32
  %op_3_b0x4_col = add i32 %morton_y, 0
  %op_3_b0x4_r_0 = add i32 %op_3_b0x4_row, 0
  %op_3_b0x4_addr_0 = mul i32 %op_3_b0x4_r_0, 16
  %op_3_b0x4_addr2_0 = add i32 %op_3_b0x4_addr_0, %op_3_b0x4_col
  %op_3_b0x4_byte_0 = mul i32 %op_3_b0x4_addr2_0, 2
  %op_3_b0x4_byte64_0 = zext i32 %op_3_b0x4_byte_0 to i64
  %op_3_b0x4_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x4_byte64_0
  %op_3_b0x4_typed_0 = bitcast i8 addrspace(3)* %op_3_b0x4_ptr_0 to half addrspace(3)*
  %op_3_b0x4_load_0 = load half, half addrspace(3)* %op_3_b0x4_typed_0
  %op_3_b0x4_r_1 = add i32 %op_3_b0x4_row, 1
  %op_3_b0x4_addr_1 = mul i32 %op_3_b0x4_r_1, 16
  %op_3_b0x4_addr2_1 = add i32 %op_3_b0x4_addr_1, %op_3_b0x4_col
  %op_3_b0x4_byte_1 = mul i32 %op_3_b0x4_addr2_1, 2
  %op_3_b0x4_byte64_1 = zext i32 %op_3_b0x4_byte_1 to i64
  %op_3_b0x4_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x4_byte64_1
  %op_3_b0x4_typed_1 = bitcast i8 addrspace(3)* %op_3_b0x4_ptr_1 to half addrspace(3)*
  %op_3_b0x4_load_1 = load half, half addrspace(3)* %op_3_b0x4_typed_1
  %op_3_b0x4_v2_a = insertelement <2 x half> undef, half %op_3_b0x4_load_0, i32 0
  %op_3_b0x4_v2 = insertelement <2 x half> %op_3_b0x4_v2_a, half %op_3_b0x4_load_1, i32 1
  %op_3_b0x4_sram_e0 = extractelement <2 x half> %op_3_b0x4_v2, i32 0
  %op_3_b0x4_sram_e1 = extractelement <2 x half> %op_3_b0x4_v2, i32 1
  %op_3_b0x4_sram_v0 = insertelement <64 x half> undef, half %op_3_b0x4_sram_e0, i32 0
  %op_3_b0x4_sram = insertelement <64 x half> %op_3_b0x4_sram_v0, half %op_3_b0x4_sram_e1, i32 1
  %op_3_c0x4 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a0_sram, <64 x half> %op_3_b0x4_sram, <64 x float> %op_2_c1x4) #3
  %op_3_b0x5_row = add i32 %morton_x, 40
  %op_3_b0x5_col = add i32 %morton_y, 0
  %op_3_b0x5_r_0 = add i32 %op_3_b0x5_row, 0
  %op_3_b0x5_addr_0 = mul i32 %op_3_b0x5_r_0, 16
  %op_3_b0x5_addr2_0 = add i32 %op_3_b0x5_addr_0, %op_3_b0x5_col
  %op_3_b0x5_byte_0 = mul i32 %op_3_b0x5_addr2_0, 2
  %op_3_b0x5_byte64_0 = zext i32 %op_3_b0x5_byte_0 to i64
  %op_3_b0x5_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x5_byte64_0
  %op_3_b0x5_typed_0 = bitcast i8 addrspace(3)* %op_3_b0x5_ptr_0 to half addrspace(3)*
  %op_3_b0x5_load_0 = load half, half addrspace(3)* %op_3_b0x5_typed_0
  %op_3_b0x5_r_1 = add i32 %op_3_b0x5_row, 1
  %op_3_b0x5_addr_1 = mul i32 %op_3_b0x5_r_1, 16
  %op_3_b0x5_addr2_1 = add i32 %op_3_b0x5_addr_1, %op_3_b0x5_col
  %op_3_b0x5_byte_1 = mul i32 %op_3_b0x5_addr2_1, 2
  %op_3_b0x5_byte64_1 = zext i32 %op_3_b0x5_byte_1 to i64
  %op_3_b0x5_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x5_byte64_1
  %op_3_b0x5_typed_1 = bitcast i8 addrspace(3)* %op_3_b0x5_ptr_1 to half addrspace(3)*
  %op_3_b0x5_load_1 = load half, half addrspace(3)* %op_3_b0x5_typed_1
  %op_3_b0x5_v2_a = insertelement <2 x half> undef, half %op_3_b0x5_load_0, i32 0
  %op_3_b0x5_v2 = insertelement <2 x half> %op_3_b0x5_v2_a, half %op_3_b0x5_load_1, i32 1
  %op_3_b0x5_sram_e0 = extractelement <2 x half> %op_3_b0x5_v2, i32 0
  %op_3_b0x5_sram_e1 = extractelement <2 x half> %op_3_b0x5_v2, i32 1
  %op_3_b0x5_sram_v0 = insertelement <64 x half> undef, half %op_3_b0x5_sram_e0, i32 0
  %op_3_b0x5_sram = insertelement <64 x half> %op_3_b0x5_sram_v0, half %op_3_b0x5_sram_e1, i32 1
  %op_3_c0x5 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a0_sram, <64 x half> %op_3_b0x5_sram, <64 x float> %op_2_c1x5) #3
  %op_3_b0x6_row = add i32 %morton_x, 48
  %op_3_b0x6_col = add i32 %morton_y, 0
  %op_3_b0x6_r_0 = add i32 %op_3_b0x6_row, 0
  %op_3_b0x6_addr_0 = mul i32 %op_3_b0x6_r_0, 16
  %op_3_b0x6_addr2_0 = add i32 %op_3_b0x6_addr_0, %op_3_b0x6_col
  %op_3_b0x6_byte_0 = mul i32 %op_3_b0x6_addr2_0, 2
  %op_3_b0x6_byte64_0 = zext i32 %op_3_b0x6_byte_0 to i64
  %op_3_b0x6_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x6_byte64_0
  %op_3_b0x6_typed_0 = bitcast i8 addrspace(3)* %op_3_b0x6_ptr_0 to half addrspace(3)*
  %op_3_b0x6_load_0 = load half, half addrspace(3)* %op_3_b0x6_typed_0
  %op_3_b0x6_r_1 = add i32 %op_3_b0x6_row, 1
  %op_3_b0x6_addr_1 = mul i32 %op_3_b0x6_r_1, 16
  %op_3_b0x6_addr2_1 = add i32 %op_3_b0x6_addr_1, %op_3_b0x6_col
  %op_3_b0x6_byte_1 = mul i32 %op_3_b0x6_addr2_1, 2
  %op_3_b0x6_byte64_1 = zext i32 %op_3_b0x6_byte_1 to i64
  %op_3_b0x6_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x6_byte64_1
  %op_3_b0x6_typed_1 = bitcast i8 addrspace(3)* %op_3_b0x6_ptr_1 to half addrspace(3)*
  %op_3_b0x6_load_1 = load half, half addrspace(3)* %op_3_b0x6_typed_1
  %op_3_b0x6_v2_a = insertelement <2 x half> undef, half %op_3_b0x6_load_0, i32 0
  %op_3_b0x6_v2 = insertelement <2 x half> %op_3_b0x6_v2_a, half %op_3_b0x6_load_1, i32 1
  %op_3_b0x6_sram_e0 = extractelement <2 x half> %op_3_b0x6_v2, i32 0
  %op_3_b0x6_sram_e1 = extractelement <2 x half> %op_3_b0x6_v2, i32 1
  %op_3_b0x6_sram_v0 = insertelement <64 x half> undef, half %op_3_b0x6_sram_e0, i32 0
  %op_3_b0x6_sram = insertelement <64 x half> %op_3_b0x6_sram_v0, half %op_3_b0x6_sram_e1, i32 1
  %op_3_c0x6 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a0_sram, <64 x half> %op_3_b0x6_sram, <64 x float> %op_2_c1x6) #3
  %op_3_b0x7_row = add i32 %morton_x, 56
  %op_3_b0x7_col = add i32 %morton_y, 0
  %op_3_b0x7_r_0 = add i32 %op_3_b0x7_row, 0
  %op_3_b0x7_addr_0 = mul i32 %op_3_b0x7_r_0, 16
  %op_3_b0x7_addr2_0 = add i32 %op_3_b0x7_addr_0, %op_3_b0x7_col
  %op_3_b0x7_byte_0 = mul i32 %op_3_b0x7_addr2_0, 2
  %op_3_b0x7_byte64_0 = zext i32 %op_3_b0x7_byte_0 to i64
  %op_3_b0x7_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x7_byte64_0
  %op_3_b0x7_typed_0 = bitcast i8 addrspace(3)* %op_3_b0x7_ptr_0 to half addrspace(3)*
  %op_3_b0x7_load_0 = load half, half addrspace(3)* %op_3_b0x7_typed_0
  %op_3_b0x7_r_1 = add i32 %op_3_b0x7_row, 1
  %op_3_b0x7_addr_1 = mul i32 %op_3_b0x7_r_1, 16
  %op_3_b0x7_addr2_1 = add i32 %op_3_b0x7_addr_1, %op_3_b0x7_col
  %op_3_b0x7_byte_1 = mul i32 %op_3_b0x7_addr2_1, 2
  %op_3_b0x7_byte64_1 = zext i32 %op_3_b0x7_byte_1 to i64
  %op_3_b0x7_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b0x7_byte64_1
  %op_3_b0x7_typed_1 = bitcast i8 addrspace(3)* %op_3_b0x7_ptr_1 to half addrspace(3)*
  %op_3_b0x7_load_1 = load half, half addrspace(3)* %op_3_b0x7_typed_1
  %op_3_b0x7_v2_a = insertelement <2 x half> undef, half %op_3_b0x7_load_0, i32 0
  %op_3_b0x7_v2 = insertelement <2 x half> %op_3_b0x7_v2_a, half %op_3_b0x7_load_1, i32 1
  %op_3_b0x7_sram_e0 = extractelement <2 x half> %op_3_b0x7_v2, i32 0
  %op_3_b0x7_sram_e1 = extractelement <2 x half> %op_3_b0x7_v2, i32 1
  %op_3_b0x7_sram_v0 = insertelement <64 x half> undef, half %op_3_b0x7_sram_e0, i32 0
  %op_3_b0x7_sram = insertelement <64 x half> %op_3_b0x7_sram_v0, half %op_3_b0x7_sram_e1, i32 1
  %op_3_c0x7 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a0_sram, <64 x half> %op_3_b0x7_sram, <64 x float> %op_2_c1x7) #3
  %op_3_a1_seq = add i32 %clamped_par_off, 0
  %op_3_a1_head = add i32 %morton_x, 56
  %op_3_a1_addr = mul i32 %op_3_a1_seq, 64
  %op_3_a1_addr2 = add i32 %op_3_a1_addr, %op_3_a1_head
  %op_3_a1_byte = mul i32 %op_3_a1_addr2, 2
  %op_3_a1_byte64 = zext i32 %op_3_a1_byte to i64
  %op_3_a1_ptr = getelementptr i8, i8 addrspace(1)* %Q, i64 %op_3_a1_byte64
  %op_3_a1_typed = bitcast i8 addrspace(1)* %op_3_a1_ptr to <2 x half> addrspace(1)*
  %op_3_a1_load = load <2 x half>, <2 x half> addrspace(1)* %op_3_a1_typed, align 4
  %op_3_a1_v2 = bitcast <2 x half> %op_3_a1_load to <2 x half>
  %op_3_a1_sram_e0 = extractelement <2 x half> %op_3_a1_v2, i32 0
  %op_3_a1_sram_e1 = extractelement <2 x half> %op_3_a1_v2, i32 1
  %op_3_a1_sram_v0 = insertelement <64 x half> undef, half %op_3_a1_sram_e0, i32 0
  %op_3_a1_sram = insertelement <64 x half> %op_3_a1_sram_v0, half %op_3_a1_sram_e1, i32 1
  %op_3_b1x0_row = add i32 %morton_x, 0
  %op_3_b1x0_col = add i32 %morton_y, 8
  %op_3_b1x0_r_0 = add i32 %op_3_b1x0_row, 0
  %op_3_b1x0_addr_0 = mul i32 %op_3_b1x0_r_0, 16
  %op_3_b1x0_addr2_0 = add i32 %op_3_b1x0_addr_0, %op_3_b1x0_col
  %op_3_b1x0_byte_0 = mul i32 %op_3_b1x0_addr2_0, 2
  %op_3_b1x0_byte64_0 = zext i32 %op_3_b1x0_byte_0 to i64
  %op_3_b1x0_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x0_byte64_0
  %op_3_b1x0_typed_0 = bitcast i8 addrspace(3)* %op_3_b1x0_ptr_0 to half addrspace(3)*
  %op_3_b1x0_load_0 = load half, half addrspace(3)* %op_3_b1x0_typed_0
  %op_3_b1x0_r_1 = add i32 %op_3_b1x0_row, 1
  %op_3_b1x0_addr_1 = mul i32 %op_3_b1x0_r_1, 16
  %op_3_b1x0_addr2_1 = add i32 %op_3_b1x0_addr_1, %op_3_b1x0_col
  %op_3_b1x0_byte_1 = mul i32 %op_3_b1x0_addr2_1, 2
  %op_3_b1x0_byte64_1 = zext i32 %op_3_b1x0_byte_1 to i64
  %op_3_b1x0_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x0_byte64_1
  %op_3_b1x0_typed_1 = bitcast i8 addrspace(3)* %op_3_b1x0_ptr_1 to half addrspace(3)*
  %op_3_b1x0_load_1 = load half, half addrspace(3)* %op_3_b1x0_typed_1
  %op_3_b1x0_v2_a = insertelement <2 x half> undef, half %op_3_b1x0_load_0, i32 0
  %op_3_b1x0_v2 = insertelement <2 x half> %op_3_b1x0_v2_a, half %op_3_b1x0_load_1, i32 1
  %op_3_b1x0_sram_e0 = extractelement <2 x half> %op_3_b1x0_v2, i32 0
  %op_3_b1x0_sram_e1 = extractelement <2 x half> %op_3_b1x0_v2, i32 1
  %op_3_b1x0_sram_v0 = insertelement <64 x half> undef, half %op_3_b1x0_sram_e0, i32 0
  %op_3_b1x0_sram = insertelement <64 x half> %op_3_b1x0_sram_v0, half %op_3_b1x0_sram_e1, i32 1
  %op_3_c1x0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a1_sram, <64 x half> %op_3_b1x0_sram, <64 x float> %op_3_c0x0) #3
  %op_3_b1x1_row = add i32 %morton_x, 8
  %op_3_b1x1_col = add i32 %morton_y, 8
  %op_3_b1x1_r_0 = add i32 %op_3_b1x1_row, 0
  %op_3_b1x1_addr_0 = mul i32 %op_3_b1x1_r_0, 16
  %op_3_b1x1_addr2_0 = add i32 %op_3_b1x1_addr_0, %op_3_b1x1_col
  %op_3_b1x1_byte_0 = mul i32 %op_3_b1x1_addr2_0, 2
  %op_3_b1x1_byte64_0 = zext i32 %op_3_b1x1_byte_0 to i64
  %op_3_b1x1_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x1_byte64_0
  %op_3_b1x1_typed_0 = bitcast i8 addrspace(3)* %op_3_b1x1_ptr_0 to half addrspace(3)*
  %op_3_b1x1_load_0 = load half, half addrspace(3)* %op_3_b1x1_typed_0
  %op_3_b1x1_r_1 = add i32 %op_3_b1x1_row, 1
  %op_3_b1x1_addr_1 = mul i32 %op_3_b1x1_r_1, 16
  %op_3_b1x1_addr2_1 = add i32 %op_3_b1x1_addr_1, %op_3_b1x1_col
  %op_3_b1x1_byte_1 = mul i32 %op_3_b1x1_addr2_1, 2
  %op_3_b1x1_byte64_1 = zext i32 %op_3_b1x1_byte_1 to i64
  %op_3_b1x1_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x1_byte64_1
  %op_3_b1x1_typed_1 = bitcast i8 addrspace(3)* %op_3_b1x1_ptr_1 to half addrspace(3)*
  %op_3_b1x1_load_1 = load half, half addrspace(3)* %op_3_b1x1_typed_1
  %op_3_b1x1_v2_a = insertelement <2 x half> undef, half %op_3_b1x1_load_0, i32 0
  %op_3_b1x1_v2 = insertelement <2 x half> %op_3_b1x1_v2_a, half %op_3_b1x1_load_1, i32 1
  %op_3_b1x1_sram_e0 = extractelement <2 x half> %op_3_b1x1_v2, i32 0
  %op_3_b1x1_sram_e1 = extractelement <2 x half> %op_3_b1x1_v2, i32 1
  %op_3_b1x1_sram_v0 = insertelement <64 x half> undef, half %op_3_b1x1_sram_e0, i32 0
  %op_3_b1x1_sram = insertelement <64 x half> %op_3_b1x1_sram_v0, half %op_3_b1x1_sram_e1, i32 1
  %op_3_c1x1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a1_sram, <64 x half> %op_3_b1x1_sram, <64 x float> %op_3_c0x1) #3
  %op_3_b1x2_row = add i32 %morton_x, 16
  %op_3_b1x2_col = add i32 %morton_y, 8
  %op_3_b1x2_r_0 = add i32 %op_3_b1x2_row, 0
  %op_3_b1x2_addr_0 = mul i32 %op_3_b1x2_r_0, 16
  %op_3_b1x2_addr2_0 = add i32 %op_3_b1x2_addr_0, %op_3_b1x2_col
  %op_3_b1x2_byte_0 = mul i32 %op_3_b1x2_addr2_0, 2
  %op_3_b1x2_byte64_0 = zext i32 %op_3_b1x2_byte_0 to i64
  %op_3_b1x2_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x2_byte64_0
  %op_3_b1x2_typed_0 = bitcast i8 addrspace(3)* %op_3_b1x2_ptr_0 to half addrspace(3)*
  %op_3_b1x2_load_0 = load half, half addrspace(3)* %op_3_b1x2_typed_0
  %op_3_b1x2_r_1 = add i32 %op_3_b1x2_row, 1
  %op_3_b1x2_addr_1 = mul i32 %op_3_b1x2_r_1, 16
  %op_3_b1x2_addr2_1 = add i32 %op_3_b1x2_addr_1, %op_3_b1x2_col
  %op_3_b1x2_byte_1 = mul i32 %op_3_b1x2_addr2_1, 2
  %op_3_b1x2_byte64_1 = zext i32 %op_3_b1x2_byte_1 to i64
  %op_3_b1x2_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x2_byte64_1
  %op_3_b1x2_typed_1 = bitcast i8 addrspace(3)* %op_3_b1x2_ptr_1 to half addrspace(3)*
  %op_3_b1x2_load_1 = load half, half addrspace(3)* %op_3_b1x2_typed_1
  %op_3_b1x2_v2_a = insertelement <2 x half> undef, half %op_3_b1x2_load_0, i32 0
  %op_3_b1x2_v2 = insertelement <2 x half> %op_3_b1x2_v2_a, half %op_3_b1x2_load_1, i32 1
  %op_3_b1x2_sram_e0 = extractelement <2 x half> %op_3_b1x2_v2, i32 0
  %op_3_b1x2_sram_e1 = extractelement <2 x half> %op_3_b1x2_v2, i32 1
  %op_3_b1x2_sram_v0 = insertelement <64 x half> undef, half %op_3_b1x2_sram_e0, i32 0
  %op_3_b1x2_sram = insertelement <64 x half> %op_3_b1x2_sram_v0, half %op_3_b1x2_sram_e1, i32 1
  %op_3_c1x2 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a1_sram, <64 x half> %op_3_b1x2_sram, <64 x float> %op_3_c0x2) #3
  %op_3_b1x3_row = add i32 %morton_x, 24
  %op_3_b1x3_col = add i32 %morton_y, 8
  %op_3_b1x3_r_0 = add i32 %op_3_b1x3_row, 0
  %op_3_b1x3_addr_0 = mul i32 %op_3_b1x3_r_0, 16
  %op_3_b1x3_addr2_0 = add i32 %op_3_b1x3_addr_0, %op_3_b1x3_col
  %op_3_b1x3_byte_0 = mul i32 %op_3_b1x3_addr2_0, 2
  %op_3_b1x3_byte64_0 = zext i32 %op_3_b1x3_byte_0 to i64
  %op_3_b1x3_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x3_byte64_0
  %op_3_b1x3_typed_0 = bitcast i8 addrspace(3)* %op_3_b1x3_ptr_0 to half addrspace(3)*
  %op_3_b1x3_load_0 = load half, half addrspace(3)* %op_3_b1x3_typed_0
  %op_3_b1x3_r_1 = add i32 %op_3_b1x3_row, 1
  %op_3_b1x3_addr_1 = mul i32 %op_3_b1x3_r_1, 16
  %op_3_b1x3_addr2_1 = add i32 %op_3_b1x3_addr_1, %op_3_b1x3_col
  %op_3_b1x3_byte_1 = mul i32 %op_3_b1x3_addr2_1, 2
  %op_3_b1x3_byte64_1 = zext i32 %op_3_b1x3_byte_1 to i64
  %op_3_b1x3_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x3_byte64_1
  %op_3_b1x3_typed_1 = bitcast i8 addrspace(3)* %op_3_b1x3_ptr_1 to half addrspace(3)*
  %op_3_b1x3_load_1 = load half, half addrspace(3)* %op_3_b1x3_typed_1
  %op_3_b1x3_v2_a = insertelement <2 x half> undef, half %op_3_b1x3_load_0, i32 0
  %op_3_b1x3_v2 = insertelement <2 x half> %op_3_b1x3_v2_a, half %op_3_b1x3_load_1, i32 1
  %op_3_b1x3_sram_e0 = extractelement <2 x half> %op_3_b1x3_v2, i32 0
  %op_3_b1x3_sram_e1 = extractelement <2 x half> %op_3_b1x3_v2, i32 1
  %op_3_b1x3_sram_v0 = insertelement <64 x half> undef, half %op_3_b1x3_sram_e0, i32 0
  %op_3_b1x3_sram = insertelement <64 x half> %op_3_b1x3_sram_v0, half %op_3_b1x3_sram_e1, i32 1
  %op_3_c1x3 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a1_sram, <64 x half> %op_3_b1x3_sram, <64 x float> %op_3_c0x3) #3
  %op_3_b1x4_row = add i32 %morton_x, 32
  %op_3_b1x4_col = add i32 %morton_y, 8
  %op_3_b1x4_r_0 = add i32 %op_3_b1x4_row, 0
  %op_3_b1x4_addr_0 = mul i32 %op_3_b1x4_r_0, 16
  %op_3_b1x4_addr2_0 = add i32 %op_3_b1x4_addr_0, %op_3_b1x4_col
  %op_3_b1x4_byte_0 = mul i32 %op_3_b1x4_addr2_0, 2
  %op_3_b1x4_byte64_0 = zext i32 %op_3_b1x4_byte_0 to i64
  %op_3_b1x4_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x4_byte64_0
  %op_3_b1x4_typed_0 = bitcast i8 addrspace(3)* %op_3_b1x4_ptr_0 to half addrspace(3)*
  %op_3_b1x4_load_0 = load half, half addrspace(3)* %op_3_b1x4_typed_0
  %op_3_b1x4_r_1 = add i32 %op_3_b1x4_row, 1
  %op_3_b1x4_addr_1 = mul i32 %op_3_b1x4_r_1, 16
  %op_3_b1x4_addr2_1 = add i32 %op_3_b1x4_addr_1, %op_3_b1x4_col
  %op_3_b1x4_byte_1 = mul i32 %op_3_b1x4_addr2_1, 2
  %op_3_b1x4_byte64_1 = zext i32 %op_3_b1x4_byte_1 to i64
  %op_3_b1x4_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x4_byte64_1
  %op_3_b1x4_typed_1 = bitcast i8 addrspace(3)* %op_3_b1x4_ptr_1 to half addrspace(3)*
  %op_3_b1x4_load_1 = load half, half addrspace(3)* %op_3_b1x4_typed_1
  %op_3_b1x4_v2_a = insertelement <2 x half> undef, half %op_3_b1x4_load_0, i32 0
  %op_3_b1x4_v2 = insertelement <2 x half> %op_3_b1x4_v2_a, half %op_3_b1x4_load_1, i32 1
  %op_3_b1x4_sram_e0 = extractelement <2 x half> %op_3_b1x4_v2, i32 0
  %op_3_b1x4_sram_e1 = extractelement <2 x half> %op_3_b1x4_v2, i32 1
  %op_3_b1x4_sram_v0 = insertelement <64 x half> undef, half %op_3_b1x4_sram_e0, i32 0
  %op_3_b1x4_sram = insertelement <64 x half> %op_3_b1x4_sram_v0, half %op_3_b1x4_sram_e1, i32 1
  %op_3_c1x4 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a1_sram, <64 x half> %op_3_b1x4_sram, <64 x float> %op_3_c0x4) #3
  %op_3_b1x5_row = add i32 %morton_x, 40
  %op_3_b1x5_col = add i32 %morton_y, 8
  %op_3_b1x5_r_0 = add i32 %op_3_b1x5_row, 0
  %op_3_b1x5_addr_0 = mul i32 %op_3_b1x5_r_0, 16
  %op_3_b1x5_addr2_0 = add i32 %op_3_b1x5_addr_0, %op_3_b1x5_col
  %op_3_b1x5_byte_0 = mul i32 %op_3_b1x5_addr2_0, 2
  %op_3_b1x5_byte64_0 = zext i32 %op_3_b1x5_byte_0 to i64
  %op_3_b1x5_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x5_byte64_0
  %op_3_b1x5_typed_0 = bitcast i8 addrspace(3)* %op_3_b1x5_ptr_0 to half addrspace(3)*
  %op_3_b1x5_load_0 = load half, half addrspace(3)* %op_3_b1x5_typed_0
  %op_3_b1x5_r_1 = add i32 %op_3_b1x5_row, 1
  %op_3_b1x5_addr_1 = mul i32 %op_3_b1x5_r_1, 16
  %op_3_b1x5_addr2_1 = add i32 %op_3_b1x5_addr_1, %op_3_b1x5_col
  %op_3_b1x5_byte_1 = mul i32 %op_3_b1x5_addr2_1, 2
  %op_3_b1x5_byte64_1 = zext i32 %op_3_b1x5_byte_1 to i64
  %op_3_b1x5_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x5_byte64_1
  %op_3_b1x5_typed_1 = bitcast i8 addrspace(3)* %op_3_b1x5_ptr_1 to half addrspace(3)*
  %op_3_b1x5_load_1 = load half, half addrspace(3)* %op_3_b1x5_typed_1
  %op_3_b1x5_v2_a = insertelement <2 x half> undef, half %op_3_b1x5_load_0, i32 0
  %op_3_b1x5_v2 = insertelement <2 x half> %op_3_b1x5_v2_a, half %op_3_b1x5_load_1, i32 1
  %op_3_b1x5_sram_e0 = extractelement <2 x half> %op_3_b1x5_v2, i32 0
  %op_3_b1x5_sram_e1 = extractelement <2 x half> %op_3_b1x5_v2, i32 1
  %op_3_b1x5_sram_v0 = insertelement <64 x half> undef, half %op_3_b1x5_sram_e0, i32 0
  %op_3_b1x5_sram = insertelement <64 x half> %op_3_b1x5_sram_v0, half %op_3_b1x5_sram_e1, i32 1
  %op_3_c1x5 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a1_sram, <64 x half> %op_3_b1x5_sram, <64 x float> %op_3_c0x5) #3
  %op_3_b1x6_row = add i32 %morton_x, 48
  %op_3_b1x6_col = add i32 %morton_y, 8
  %op_3_b1x6_r_0 = add i32 %op_3_b1x6_row, 0
  %op_3_b1x6_addr_0 = mul i32 %op_3_b1x6_r_0, 16
  %op_3_b1x6_addr2_0 = add i32 %op_3_b1x6_addr_0, %op_3_b1x6_col
  %op_3_b1x6_byte_0 = mul i32 %op_3_b1x6_addr2_0, 2
  %op_3_b1x6_byte64_0 = zext i32 %op_3_b1x6_byte_0 to i64
  %op_3_b1x6_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x6_byte64_0
  %op_3_b1x6_typed_0 = bitcast i8 addrspace(3)* %op_3_b1x6_ptr_0 to half addrspace(3)*
  %op_3_b1x6_load_0 = load half, half addrspace(3)* %op_3_b1x6_typed_0
  %op_3_b1x6_r_1 = add i32 %op_3_b1x6_row, 1
  %op_3_b1x6_addr_1 = mul i32 %op_3_b1x6_r_1, 16
  %op_3_b1x6_addr2_1 = add i32 %op_3_b1x6_addr_1, %op_3_b1x6_col
  %op_3_b1x6_byte_1 = mul i32 %op_3_b1x6_addr2_1, 2
  %op_3_b1x6_byte64_1 = zext i32 %op_3_b1x6_byte_1 to i64
  %op_3_b1x6_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x6_byte64_1
  %op_3_b1x6_typed_1 = bitcast i8 addrspace(3)* %op_3_b1x6_ptr_1 to half addrspace(3)*
  %op_3_b1x6_load_1 = load half, half addrspace(3)* %op_3_b1x6_typed_1
  %op_3_b1x6_v2_a = insertelement <2 x half> undef, half %op_3_b1x6_load_0, i32 0
  %op_3_b1x6_v2 = insertelement <2 x half> %op_3_b1x6_v2_a, half %op_3_b1x6_load_1, i32 1
  %op_3_b1x6_sram_e0 = extractelement <2 x half> %op_3_b1x6_v2, i32 0
  %op_3_b1x6_sram_e1 = extractelement <2 x half> %op_3_b1x6_v2, i32 1
  %op_3_b1x6_sram_v0 = insertelement <64 x half> undef, half %op_3_b1x6_sram_e0, i32 0
  %op_3_b1x6_sram = insertelement <64 x half> %op_3_b1x6_sram_v0, half %op_3_b1x6_sram_e1, i32 1
  %op_3_c1x6 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a1_sram, <64 x half> %op_3_b1x6_sram, <64 x float> %op_3_c0x6) #3
  %op_3_b1x7_row = add i32 %morton_x, 56
  %op_3_b1x7_col = add i32 %morton_y, 8
  %op_3_b1x7_r_0 = add i32 %op_3_b1x7_row, 0
  %op_3_b1x7_addr_0 = mul i32 %op_3_b1x7_r_0, 16
  %op_3_b1x7_addr2_0 = add i32 %op_3_b1x7_addr_0, %op_3_b1x7_col
  %op_3_b1x7_byte_0 = mul i32 %op_3_b1x7_addr2_0, 2
  %op_3_b1x7_byte64_0 = zext i32 %op_3_b1x7_byte_0 to i64
  %op_3_b1x7_ptr_0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x7_byte64_0
  %op_3_b1x7_typed_0 = bitcast i8 addrspace(3)* %op_3_b1x7_ptr_0 to half addrspace(3)*
  %op_3_b1x7_load_0 = load half, half addrspace(3)* %op_3_b1x7_typed_0
  %op_3_b1x7_r_1 = add i32 %op_3_b1x7_row, 1
  %op_3_b1x7_addr_1 = mul i32 %op_3_b1x7_r_1, 16
  %op_3_b1x7_addr2_1 = add i32 %op_3_b1x7_addr_1, %op_3_b1x7_col
  %op_3_b1x7_byte_1 = mul i32 %op_3_b1x7_addr2_1, 2
  %op_3_b1x7_byte64_1 = zext i32 %op_3_b1x7_byte_1 to i64
  %op_3_b1x7_ptr_1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %op_3_b1x7_byte64_1
  %op_3_b1x7_typed_1 = bitcast i8 addrspace(3)* %op_3_b1x7_ptr_1 to half addrspace(3)*
  %op_3_b1x7_load_1 = load half, half addrspace(3)* %op_3_b1x7_typed_1
  %op_3_b1x7_v2_a = insertelement <2 x half> undef, half %op_3_b1x7_load_0, i32 0
  %op_3_b1x7_v2 = insertelement <2 x half> %op_3_b1x7_v2_a, half %op_3_b1x7_load_1, i32 1
  %op_3_b1x7_sram_e0 = extractelement <2 x half> %op_3_b1x7_v2, i32 0
  %op_3_b1x7_sram_e1 = extractelement <2 x half> %op_3_b1x7_v2, i32 1
  %op_3_b1x7_sram_v0 = insertelement <64 x half> undef, half %op_3_b1x7_sram_e0, i32 0
  %op_3_b1x7_sram = insertelement <64 x half> %op_3_b1x7_sram_v0, half %op_3_b1x7_sram_e1, i32 1
  %op_3_c1x7 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %op_3_a1_sram, <64 x half> %op_3_b1x7_sram, <64 x float> %op_3_c0x7) #3
  call void @air.wg.barrier(i32 2, i32 1)

  %s_0 = bitcast <64 x float> %op_3_c1x0 to <64 x float>
  %s_1 = bitcast <64 x float> %op_3_c1x1 to <64 x float>
  %s_2 = bitcast <64 x float> %op_3_c1x2 to <64 x float>
  %s_3 = bitcast <64 x float> %op_3_c1x3 to <64 x float>
  %s_4 = bitcast <64 x float> %op_3_c1x4 to <64 x float>
  %s_5 = bitcast <64 x float> %op_3_c1x5 to <64 x float>
  %s_6 = bitcast <64 x float> %op_3_c1x6 to <64 x float>
  %s_7 = bitcast <64 x float> %op_3_c1x7 to <64 x float>
  ; === Wait for V[c, d_outer=0] copy ===
  ; Sync wait (wv_) — barrier only (data already in TG)
  call void @air.wg.barrier(i32 2, i32 1)
  br label %wv_after_wait

wv_after_wait:
  ; === Mask attention matrix edge ===
  ; No masking needed
  %me_s_0 = bitcast <64 x float> %s_0 to <64 x float>
  %me_s_1 = bitcast <64 x float> %s_1 to <64 x float>
  %me_s_2 = bitcast <64 x float> %s_2 to <64 x float>
  %me_s_3 = bitcast <64 x float> %s_3 to <64 x float>
  %me_s_4 = bitcast <64 x float> %s_4 to <64 x float>
  %me_s_5 = bitcast <64 x float> %s_5 to <64 x float>
  %me_s_6 = bitcast <64 x float> %s_6 to <64 x float>
  %me_s_7 = bitcast <64 x float> %s_7 to <64 x float>
  ; === Reduce max ===
  %rm_e0_0 = extractelement <64 x float> %me_s_0, i32 0
  %rm_e1_0 = extractelement <64 x float> %me_s_0, i32 1
  %rm_max_0 = fcmp fast ogt float %rm_e0_0, %rm_e1_0
  %rm_m_0 = select i1 %rm_max_0, float %rm_e0_0, float %rm_e1_0
  %rm_e0_1 = extractelement <64 x float> %me_s_1, i32 0
  %rm_e1_1 = extractelement <64 x float> %me_s_1, i32 1
  %rm_cmp0_1 = fcmp fast ogt float %rm_e0_1, %rm_m_0
  %rm_sel0_1 = select i1 %rm_cmp0_1, float %rm_e0_1, float %rm_m_0
  %rm_cmp1_1 = fcmp fast ogt float %rm_e1_1, %rm_sel0_1
  %rm_m_1 = select i1 %rm_cmp1_1, float %rm_e1_1, float %rm_sel0_1
  %rm_e0_2 = extractelement <64 x float> %me_s_2, i32 0
  %rm_e1_2 = extractelement <64 x float> %me_s_2, i32 1
  %rm_cmp0_2 = fcmp fast ogt float %rm_e0_2, %rm_m_1
  %rm_sel0_2 = select i1 %rm_cmp0_2, float %rm_e0_2, float %rm_m_1
  %rm_cmp1_2 = fcmp fast ogt float %rm_e1_2, %rm_sel0_2
  %rm_m_2 = select i1 %rm_cmp1_2, float %rm_e1_2, float %rm_sel0_2
  %rm_e0_3 = extractelement <64 x float> %me_s_3, i32 0
  %rm_e1_3 = extractelement <64 x float> %me_s_3, i32 1
  %rm_cmp0_3 = fcmp fast ogt float %rm_e0_3, %rm_m_2
  %rm_sel0_3 = select i1 %rm_cmp0_3, float %rm_e0_3, float %rm_m_2
  %rm_cmp1_3 = fcmp fast ogt float %rm_e1_3, %rm_sel0_3
  %rm_m_3 = select i1 %rm_cmp1_3, float %rm_e1_3, float %rm_sel0_3
  %rm_e0_4 = extractelement <64 x float> %me_s_4, i32 0
  %rm_e1_4 = extractelement <64 x float> %me_s_4, i32 1
  %rm_cmp0_4 = fcmp fast ogt float %rm_e0_4, %rm_m_3
  %rm_sel0_4 = select i1 %rm_cmp0_4, float %rm_e0_4, float %rm_m_3
  %rm_cmp1_4 = fcmp fast ogt float %rm_e1_4, %rm_sel0_4
  %rm_m_4 = select i1 %rm_cmp1_4, float %rm_e1_4, float %rm_sel0_4
  %rm_e0_5 = extractelement <64 x float> %me_s_5, i32 0
  %rm_e1_5 = extractelement <64 x float> %me_s_5, i32 1
  %rm_cmp0_5 = fcmp fast ogt float %rm_e0_5, %rm_m_4
  %rm_sel0_5 = select i1 %rm_cmp0_5, float %rm_e0_5, float %rm_m_4
  %rm_cmp1_5 = fcmp fast ogt float %rm_e1_5, %rm_sel0_5
  %rm_m_5 = select i1 %rm_cmp1_5, float %rm_e1_5, float %rm_sel0_5
  %rm_e0_6 = extractelement <64 x float> %me_s_6, i32 0
  %rm_e1_6 = extractelement <64 x float> %me_s_6, i32 1
  %rm_cmp0_6 = fcmp fast ogt float %rm_e0_6, %rm_m_5
  %rm_sel0_6 = select i1 %rm_cmp0_6, float %rm_e0_6, float %rm_m_5
  %rm_cmp1_6 = fcmp fast ogt float %rm_e1_6, %rm_sel0_6
  %rm_m_6 = select i1 %rm_cmp1_6, float %rm_e1_6, float %rm_sel0_6
  %rm_e0_7 = extractelement <64 x float> %me_s_7, i32 0
  %rm_e1_7 = extractelement <64 x float> %me_s_7, i32 1
  %rm_cmp0_7 = fcmp fast ogt float %rm_e0_7, %rm_m_6
  %rm_sel0_7 = select i1 %rm_cmp0_7, float %rm_e0_7, float %rm_m_6
  %rm_cmp1_7 = fcmp fast ogt float %rm_e1_7, %rm_sel0_7
  %rm_m_7 = select i1 %rm_cmp1_7, float %rm_e1_7, float %rm_sel0_7
  %rm_mf = bitcast float %rm_m_7 to float
  %rm_shuf1 = call float @air.simd_shuffle_xor.f32(float %rm_mf, i32 1)
  %rm_cmp_s1 = fcmp fast ogt float %rm_mf, %rm_shuf1
  %rm_max_s1 = select i1 %rm_cmp_s1, float %rm_mf, float %rm_shuf1
  %rm_shuf8 = call float @air.simd_shuffle_xor.f32(float %rm_max_s1, i32 8)
  %rm_cmp_s8 = fcmp fast ogt float %rm_max_s1, %rm_shuf8
  %rm_m_new = select i1 %rm_cmp_s8, float %rm_max_s1, float %rm_shuf8
  %rm_m_new_scaled = fmul fast float %rm_m_new, 0x3FC7154760000000
  ; === Correct O ===
  %co_m_gt = fcmp fast ogt float %rm_m_new_scaled, %m_phi
  %co_m_diff = fsub fast float %m_phi, %rm_m_new_scaled
  %co_exp_diff = call fast float @llvm.exp2.f32(float %co_m_diff)
  %co_correction = select i1 %co_m_gt, float %co_exp_diff, float 1.0
  %co_m_upd = select i1 %co_m_gt, float %rm_m_new_scaled, float %m_phi
  ; === Compute P = exp2(S * scale - m) ===
  %sp_s0_0 = extractelement <64 x float> %me_s_0, i32 0
  %sp_s1_0 = extractelement <64 x float> %me_s_0, i32 1
  %sp_scaled0_0 = fmul fast float %sp_s0_0, 0x3FC7154760000000
  %sp_shifted0_0 = fsub fast float %sp_scaled0_0, %co_m_upd
  %sp_p0f_0 = call fast float @llvm.exp2.f32(float %sp_shifted0_0)
  %sp_scaled1_0 = fmul fast float %sp_s1_0, 0x3FC7154760000000
  %sp_shifted1_0 = fsub fast float %sp_scaled1_0, %co_m_upd
  %sp_p1f_0 = call fast float @llvm.exp2.f32(float %sp_shifted1_0)
  %sp_pv0_0 = insertelement <64 x float> undef, float %sp_p0f_0, i32 0
  %sp_p_0 = insertelement <64 x float> %sp_pv0_0, float %sp_p1f_0, i32 1
  %sp_s0_1 = extractelement <64 x float> %me_s_1, i32 0
  %sp_s1_1 = extractelement <64 x float> %me_s_1, i32 1
  %sp_scaled0_1 = fmul fast float %sp_s0_1, 0x3FC7154760000000
  %sp_shifted0_1 = fsub fast float %sp_scaled0_1, %co_m_upd
  %sp_p0f_1 = call fast float @llvm.exp2.f32(float %sp_shifted0_1)
  %sp_scaled1_1 = fmul fast float %sp_s1_1, 0x3FC7154760000000
  %sp_shifted1_1 = fsub fast float %sp_scaled1_1, %co_m_upd
  %sp_p1f_1 = call fast float @llvm.exp2.f32(float %sp_shifted1_1)
  %sp_pv0_1 = insertelement <64 x float> undef, float %sp_p0f_1, i32 0
  %sp_p_1 = insertelement <64 x float> %sp_pv0_1, float %sp_p1f_1, i32 1
  %sp_s0_2 = extractelement <64 x float> %me_s_2, i32 0
  %sp_s1_2 = extractelement <64 x float> %me_s_2, i32 1
  %sp_scaled0_2 = fmul fast float %sp_s0_2, 0x3FC7154760000000
  %sp_shifted0_2 = fsub fast float %sp_scaled0_2, %co_m_upd
  %sp_p0f_2 = call fast float @llvm.exp2.f32(float %sp_shifted0_2)
  %sp_scaled1_2 = fmul fast float %sp_s1_2, 0x3FC7154760000000
  %sp_shifted1_2 = fsub fast float %sp_scaled1_2, %co_m_upd
  %sp_p1f_2 = call fast float @llvm.exp2.f32(float %sp_shifted1_2)
  %sp_pv0_2 = insertelement <64 x float> undef, float %sp_p0f_2, i32 0
  %sp_p_2 = insertelement <64 x float> %sp_pv0_2, float %sp_p1f_2, i32 1
  %sp_s0_3 = extractelement <64 x float> %me_s_3, i32 0
  %sp_s1_3 = extractelement <64 x float> %me_s_3, i32 1
  %sp_scaled0_3 = fmul fast float %sp_s0_3, 0x3FC7154760000000
  %sp_shifted0_3 = fsub fast float %sp_scaled0_3, %co_m_upd
  %sp_p0f_3 = call fast float @llvm.exp2.f32(float %sp_shifted0_3)
  %sp_scaled1_3 = fmul fast float %sp_s1_3, 0x3FC7154760000000
  %sp_shifted1_3 = fsub fast float %sp_scaled1_3, %co_m_upd
  %sp_p1f_3 = call fast float @llvm.exp2.f32(float %sp_shifted1_3)
  %sp_pv0_3 = insertelement <64 x float> undef, float %sp_p0f_3, i32 0
  %sp_p_3 = insertelement <64 x float> %sp_pv0_3, float %sp_p1f_3, i32 1
  %sp_s0_4 = extractelement <64 x float> %me_s_4, i32 0
  %sp_s1_4 = extractelement <64 x float> %me_s_4, i32 1
  %sp_scaled0_4 = fmul fast float %sp_s0_4, 0x3FC7154760000000
  %sp_shifted0_4 = fsub fast float %sp_scaled0_4, %co_m_upd
  %sp_p0f_4 = call fast float @llvm.exp2.f32(float %sp_shifted0_4)
  %sp_scaled1_4 = fmul fast float %sp_s1_4, 0x3FC7154760000000
  %sp_shifted1_4 = fsub fast float %sp_scaled1_4, %co_m_upd
  %sp_p1f_4 = call fast float @llvm.exp2.f32(float %sp_shifted1_4)
  %sp_pv0_4 = insertelement <64 x float> undef, float %sp_p0f_4, i32 0
  %sp_p_4 = insertelement <64 x float> %sp_pv0_4, float %sp_p1f_4, i32 1
  %sp_s0_5 = extractelement <64 x float> %me_s_5, i32 0
  %sp_s1_5 = extractelement <64 x float> %me_s_5, i32 1
  %sp_scaled0_5 = fmul fast float %sp_s0_5, 0x3FC7154760000000
  %sp_shifted0_5 = fsub fast float %sp_scaled0_5, %co_m_upd
  %sp_p0f_5 = call fast float @llvm.exp2.f32(float %sp_shifted0_5)
  %sp_scaled1_5 = fmul fast float %sp_s1_5, 0x3FC7154760000000
  %sp_shifted1_5 = fsub fast float %sp_scaled1_5, %co_m_upd
  %sp_p1f_5 = call fast float @llvm.exp2.f32(float %sp_shifted1_5)
  %sp_pv0_5 = insertelement <64 x float> undef, float %sp_p0f_5, i32 0
  %sp_p_5 = insertelement <64 x float> %sp_pv0_5, float %sp_p1f_5, i32 1
  %sp_s0_6 = extractelement <64 x float> %me_s_6, i32 0
  %sp_s1_6 = extractelement <64 x float> %me_s_6, i32 1
  %sp_scaled0_6 = fmul fast float %sp_s0_6, 0x3FC7154760000000
  %sp_shifted0_6 = fsub fast float %sp_scaled0_6, %co_m_upd
  %sp_p0f_6 = call fast float @llvm.exp2.f32(float %sp_shifted0_6)
  %sp_scaled1_6 = fmul fast float %sp_s1_6, 0x3FC7154760000000
  %sp_shifted1_6 = fsub fast float %sp_scaled1_6, %co_m_upd
  %sp_p1f_6 = call fast float @llvm.exp2.f32(float %sp_shifted1_6)
  %sp_pv0_6 = insertelement <64 x float> undef, float %sp_p0f_6, i32 0
  %sp_p_6 = insertelement <64 x float> %sp_pv0_6, float %sp_p1f_6, i32 1
  %sp_s0_7 = extractelement <64 x float> %me_s_7, i32 0
  %sp_s1_7 = extractelement <64 x float> %me_s_7, i32 1
  %sp_scaled0_7 = fmul fast float %sp_s0_7, 0x3FC7154760000000
  %sp_shifted0_7 = fsub fast float %sp_scaled0_7, %co_m_upd
  %sp_p0f_7 = call fast float @llvm.exp2.f32(float %sp_shifted0_7)
  %sp_scaled1_7 = fmul fast float %sp_s1_7, 0x3FC7154760000000
  %sp_shifted1_7 = fsub fast float %sp_scaled1_7, %co_m_upd
  %sp_p1f_7 = call fast float @llvm.exp2.f32(float %sp_shifted1_7)
  %sp_pv0_7 = insertelement <64 x float> undef, float %sp_p0f_7, i32 0
  %sp_p_7 = insertelement <64 x float> %sp_pv0_7, float %sp_p1f_7, i32 1
  ; === Reduce sum ===
  %rs_e0_0 = extractelement <64 x float> %sp_p_0, i32 0
  %rs_e1_0 = extractelement <64 x float> %sp_p_0, i32 1
  %rs_sum_0 = fadd fast float %rs_e0_0, %rs_e1_0
  %rs_e0_1 = extractelement <64 x float> %sp_p_1, i32 0
  %rs_e1_1 = extractelement <64 x float> %sp_p_1, i32 1
  %rs_add0_1 = fadd fast float %rs_sum_0, %rs_e0_1
  %rs_sum_1 = fadd fast float %rs_add0_1, %rs_e1_1
  %rs_e0_2 = extractelement <64 x float> %sp_p_2, i32 0
  %rs_e1_2 = extractelement <64 x float> %sp_p_2, i32 1
  %rs_add0_2 = fadd fast float %rs_sum_1, %rs_e0_2
  %rs_sum_2 = fadd fast float %rs_add0_2, %rs_e1_2
  %rs_e0_3 = extractelement <64 x float> %sp_p_3, i32 0
  %rs_e1_3 = extractelement <64 x float> %sp_p_3, i32 1
  %rs_add0_3 = fadd fast float %rs_sum_2, %rs_e0_3
  %rs_sum_3 = fadd fast float %rs_add0_3, %rs_e1_3
  %rs_e0_4 = extractelement <64 x float> %sp_p_4, i32 0
  %rs_e1_4 = extractelement <64 x float> %sp_p_4, i32 1
  %rs_add0_4 = fadd fast float %rs_sum_3, %rs_e0_4
  %rs_sum_4 = fadd fast float %rs_add0_4, %rs_e1_4
  %rs_e0_5 = extractelement <64 x float> %sp_p_5, i32 0
  %rs_e1_5 = extractelement <64 x float> %sp_p_5, i32 1
  %rs_add0_5 = fadd fast float %rs_sum_4, %rs_e0_5
  %rs_sum_5 = fadd fast float %rs_add0_5, %rs_e1_5
  %rs_e0_6 = extractelement <64 x float> %sp_p_6, i32 0
  %rs_e1_6 = extractelement <64 x float> %sp_p_6, i32 1
  %rs_add0_6 = fadd fast float %rs_sum_5, %rs_e0_6
  %rs_sum_6 = fadd fast float %rs_add0_6, %rs_e1_6
  %rs_e0_7 = extractelement <64 x float> %sp_p_7, i32 0
  %rs_e1_7 = extractelement <64 x float> %sp_p_7, i32 1
  %rs_add0_7 = fadd fast float %rs_sum_6, %rs_e0_7
  %rs_sum_7 = fadd fast float %rs_add0_7, %rs_e1_7
  %rs_shuf1 = call float @air.simd_shuffle_xor.f32(float %rs_sum_7, i32 1)
  %rs_sum_s1 = fadd fast float %rs_sum_7, %rs_shuf1
  %rs_shuf8 = call float @air.simd_shuffle_xor.f32(float %rs_sum_s1, i32 8)
  %rs_l_new_part = fadd fast float %rs_sum_s1, %rs_shuf8
  %rs_l_corrected = fmul fast float %l_phi, %co_correction
  %rs_l_new = fadd fast float %rs_l_corrected, %rs_l_new_part
  ; === Accumulate o += P * V ===
  %acc_scale_e0_0 = extractelement <64 x float> %o_phi_0, i32 0
  %acc_scale_e1_0 = extractelement <64 x float> %o_phi_0, i32 1
  %acc_scaled_e0_0 = fmul fast float %acc_scale_e0_0, %co_correction
  %acc_scaled_e1_0 = fmul fast float %acc_scale_e1_0, %co_correction
  %acc_sv0_0 = insertelement <64 x float> %o_phi_0, float %acc_scaled_e0_0, i32 0
  %acc_corrected_0 = insertelement <64 x float> %acc_sv0_0, float %acc_scaled_e1_0, i32 1
  %acc_scale_e0_1 = extractelement <64 x float> %o_phi_1, i32 0
  %acc_scale_e1_1 = extractelement <64 x float> %o_phi_1, i32 1
  %acc_scaled_e0_1 = fmul fast float %acc_scale_e0_1, %co_correction
  %acc_scaled_e1_1 = fmul fast float %acc_scale_e1_1, %co_correction
  %acc_sv0_1 = insertelement <64 x float> %o_phi_1, float %acc_scaled_e0_1, i32 0
  %acc_corrected_1 = insertelement <64 x float> %acc_sv0_1, float %acc_scaled_e1_1, i32 1
  %acc_scale_e0_2 = extractelement <64 x float> %o_phi_2, i32 0
  %acc_scale_e1_2 = extractelement <64 x float> %o_phi_2, i32 1
  %acc_scaled_e0_2 = fmul fast float %acc_scale_e0_2, %co_correction
  %acc_scaled_e1_2 = fmul fast float %acc_scale_e1_2, %co_correction
  %acc_sv0_2 = insertelement <64 x float> %o_phi_2, float %acc_scaled_e0_2, i32 0
  %acc_corrected_2 = insertelement <64 x float> %acc_sv0_2, float %acc_scaled_e1_2, i32 1
  %acc_scale_e0_3 = extractelement <64 x float> %o_phi_3, i32 0
  %acc_scale_e1_3 = extractelement <64 x float> %o_phi_3, i32 1
  %acc_scaled_e0_3 = fmul fast float %acc_scale_e0_3, %co_correction
  %acc_scaled_e1_3 = fmul fast float %acc_scale_e1_3, %co_correction
  %acc_sv0_3 = insertelement <64 x float> %o_phi_3, float %acc_scaled_e0_3, i32 0
  %acc_corrected_3 = insertelement <64 x float> %acc_sv0_3, float %acc_scaled_e1_3, i32 1
  %acc_scale_e0_4 = extractelement <64 x float> %o_phi_4, i32 0
  %acc_scale_e1_4 = extractelement <64 x float> %o_phi_4, i32 1
  %acc_scaled_e0_4 = fmul fast float %acc_scale_e0_4, %co_correction
  %acc_scaled_e1_4 = fmul fast float %acc_scale_e1_4, %co_correction
  %acc_sv0_4 = insertelement <64 x float> %o_phi_4, float %acc_scaled_e0_4, i32 0
  %acc_corrected_4 = insertelement <64 x float> %acc_sv0_4, float %acc_scaled_e1_4, i32 1
  %acc_scale_e0_5 = extractelement <64 x float> %o_phi_5, i32 0
  %acc_scale_e1_5 = extractelement <64 x float> %o_phi_5, i32 1
  %acc_scaled_e0_5 = fmul fast float %acc_scale_e0_5, %co_correction
  %acc_scaled_e1_5 = fmul fast float %acc_scale_e1_5, %co_correction
  %acc_sv0_5 = insertelement <64 x float> %o_phi_5, float %acc_scaled_e0_5, i32 0
  %acc_corrected_5 = insertelement <64 x float> %acc_sv0_5, float %acc_scaled_e1_5, i32 1
  %acc_scale_e0_6 = extractelement <64 x float> %o_phi_6, i32 0
  %acc_scale_e1_6 = extractelement <64 x float> %o_phi_6, i32 1
  %acc_scaled_e0_6 = fmul fast float %acc_scale_e0_6, %co_correction
  %acc_scaled_e1_6 = fmul fast float %acc_scale_e1_6, %co_correction
  %acc_sv0_6 = insertelement <64 x float> %o_phi_6, float %acc_scaled_e0_6, i32 0
  %acc_corrected_6 = insertelement <64 x float> %acc_sv0_6, float %acc_scaled_e1_6, i32 1
  %acc_scale_e0_7 = extractelement <64 x float> %o_phi_7, i32 0
  %acc_scale_e1_7 = extractelement <64 x float> %o_phi_7, i32 1
  %acc_scaled_e0_7 = fmul fast float %acc_scale_e0_7, %co_correction
  %acc_scaled_e1_7 = fmul fast float %acc_scale_e1_7, %co_correction
  %acc_sv0_7 = insertelement <64 x float> %o_phi_7, float %acc_scaled_e0_7, i32 0
  %acc_corrected_7 = insertelement <64 x float> %acc_sv0_7, float %acc_scaled_e1_7, i32 1
  %acc_0_v0x0_row = add i32 %morton_y, 0
  %acc_0_v0x0_col = add i32 %morton_x, 0
  %acc_0_v0x0_addr = mul i32 %acc_0_v0x0_row, 16
  %acc_0_v0x0_addr2 = add i32 %acc_0_v0x0_addr, %acc_0_v0x0_col
  %acc_0_v0x0_byte = mul i32 %acc_0_v0x0_addr2, 2
  %acc_0_v0x0_byte64 = zext i32 %acc_0_v0x0_byte to i64
  %acc_0_v0x0_byte64o = add i64 %acc_0_v0x0_byte64, 2048
  %acc_0_v0x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v0x0_byte64o
  %acc_0_v0x0_typed = bitcast i8 addrspace(3)* %acc_0_v0x0_ptr to <2 x half> addrspace(3)*
  %acc_0_v0x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v0x0_typed, align 4
  %acc_0_v0x0_v2 = bitcast <2 x half> %acc_0_v0x0_load to <2 x half>
  %acc_0_v0x0_sram_e0 = extractelement <2 x half> %acc_0_v0x0_v2, i32 0
  %acc_0_v0x0_sram_e1 = extractelement <2 x half> %acc_0_v0x0_v2, i32 1
  %acc_0_v0x0_sram_v0 = insertelement <64 x half> undef, half %acc_0_v0x0_sram_e0, i32 0
  %acc_0_v0x0_sram = insertelement <64 x half> %acc_0_v0x0_sram_v0, half %acc_0_v0x0_sram_e1, i32 1
  %acc_0_c0d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_0, <64 x half> %acc_0_v0x0_sram, <64 x float> %acc_corrected_0) #3
  %acc_0_v0x1_row = add i32 %morton_y, 0
  %acc_0_v0x1_col = add i32 %morton_x, 8
  %acc_0_v0x1_addr = mul i32 %acc_0_v0x1_row, 16
  %acc_0_v0x1_addr2 = add i32 %acc_0_v0x1_addr, %acc_0_v0x1_col
  %acc_0_v0x1_byte = mul i32 %acc_0_v0x1_addr2, 2
  %acc_0_v0x1_byte64 = zext i32 %acc_0_v0x1_byte to i64
  %acc_0_v0x1_byte64o = add i64 %acc_0_v0x1_byte64, 2048
  %acc_0_v0x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v0x1_byte64o
  %acc_0_v0x1_typed = bitcast i8 addrspace(3)* %acc_0_v0x1_ptr to <2 x half> addrspace(3)*
  %acc_0_v0x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v0x1_typed, align 4
  %acc_0_v0x1_v2 = bitcast <2 x half> %acc_0_v0x1_load to <2 x half>
  %acc_0_v0x1_sram_e0 = extractelement <2 x half> %acc_0_v0x1_v2, i32 0
  %acc_0_v0x1_sram_e1 = extractelement <2 x half> %acc_0_v0x1_v2, i32 1
  %acc_0_v0x1_sram_v0 = insertelement <64 x half> undef, half %acc_0_v0x1_sram_e0, i32 0
  %acc_0_v0x1_sram = insertelement <64 x half> %acc_0_v0x1_sram_v0, half %acc_0_v0x1_sram_e1, i32 1
  %acc_0_c0d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_0, <64 x half> %acc_0_v0x1_sram, <64 x float> %acc_corrected_1) #3
  %acc_0_v1x0_row = add i32 %morton_y, 8
  %acc_0_v1x0_col = add i32 %morton_x, 0
  %acc_0_v1x0_addr = mul i32 %acc_0_v1x0_row, 16
  %acc_0_v1x0_addr2 = add i32 %acc_0_v1x0_addr, %acc_0_v1x0_col
  %acc_0_v1x0_byte = mul i32 %acc_0_v1x0_addr2, 2
  %acc_0_v1x0_byte64 = zext i32 %acc_0_v1x0_byte to i64
  %acc_0_v1x0_byte64o = add i64 %acc_0_v1x0_byte64, 2048
  %acc_0_v1x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v1x0_byte64o
  %acc_0_v1x0_typed = bitcast i8 addrspace(3)* %acc_0_v1x0_ptr to <2 x half> addrspace(3)*
  %acc_0_v1x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v1x0_typed, align 4
  %acc_0_v1x0_v2 = bitcast <2 x half> %acc_0_v1x0_load to <2 x half>
  %acc_0_v1x0_sram_e0 = extractelement <2 x half> %acc_0_v1x0_v2, i32 0
  %acc_0_v1x0_sram_e1 = extractelement <2 x half> %acc_0_v1x0_v2, i32 1
  %acc_0_v1x0_sram_v0 = insertelement <64 x half> undef, half %acc_0_v1x0_sram_e0, i32 0
  %acc_0_v1x0_sram = insertelement <64 x half> %acc_0_v1x0_sram_v0, half %acc_0_v1x0_sram_e1, i32 1
  %acc_0_c1d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_1, <64 x half> %acc_0_v1x0_sram, <64 x float> %acc_0_c0d0) #3
  %acc_0_v1x1_row = add i32 %morton_y, 8
  %acc_0_v1x1_col = add i32 %morton_x, 8
  %acc_0_v1x1_addr = mul i32 %acc_0_v1x1_row, 16
  %acc_0_v1x1_addr2 = add i32 %acc_0_v1x1_addr, %acc_0_v1x1_col
  %acc_0_v1x1_byte = mul i32 %acc_0_v1x1_addr2, 2
  %acc_0_v1x1_byte64 = zext i32 %acc_0_v1x1_byte to i64
  %acc_0_v1x1_byte64o = add i64 %acc_0_v1x1_byte64, 2048
  %acc_0_v1x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v1x1_byte64o
  %acc_0_v1x1_typed = bitcast i8 addrspace(3)* %acc_0_v1x1_ptr to <2 x half> addrspace(3)*
  %acc_0_v1x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v1x1_typed, align 4
  %acc_0_v1x1_v2 = bitcast <2 x half> %acc_0_v1x1_load to <2 x half>
  %acc_0_v1x1_sram_e0 = extractelement <2 x half> %acc_0_v1x1_v2, i32 0
  %acc_0_v1x1_sram_e1 = extractelement <2 x half> %acc_0_v1x1_v2, i32 1
  %acc_0_v1x1_sram_v0 = insertelement <64 x half> undef, half %acc_0_v1x1_sram_e0, i32 0
  %acc_0_v1x1_sram = insertelement <64 x half> %acc_0_v1x1_sram_v0, half %acc_0_v1x1_sram_e1, i32 1
  %acc_0_c1d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_1, <64 x half> %acc_0_v1x1_sram, <64 x float> %acc_0_c0d1) #3
  %acc_0_v2x0_row = add i32 %morton_y, 16
  %acc_0_v2x0_col = add i32 %morton_x, 0
  %acc_0_v2x0_addr = mul i32 %acc_0_v2x0_row, 16
  %acc_0_v2x0_addr2 = add i32 %acc_0_v2x0_addr, %acc_0_v2x0_col
  %acc_0_v2x0_byte = mul i32 %acc_0_v2x0_addr2, 2
  %acc_0_v2x0_byte64 = zext i32 %acc_0_v2x0_byte to i64
  %acc_0_v2x0_byte64o = add i64 %acc_0_v2x0_byte64, 2048
  %acc_0_v2x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v2x0_byte64o
  %acc_0_v2x0_typed = bitcast i8 addrspace(3)* %acc_0_v2x0_ptr to <2 x half> addrspace(3)*
  %acc_0_v2x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v2x0_typed, align 4
  %acc_0_v2x0_v2 = bitcast <2 x half> %acc_0_v2x0_load to <2 x half>
  %acc_0_v2x0_sram_e0 = extractelement <2 x half> %acc_0_v2x0_v2, i32 0
  %acc_0_v2x0_sram_e1 = extractelement <2 x half> %acc_0_v2x0_v2, i32 1
  %acc_0_v2x0_sram_v0 = insertelement <64 x half> undef, half %acc_0_v2x0_sram_e0, i32 0
  %acc_0_v2x0_sram = insertelement <64 x half> %acc_0_v2x0_sram_v0, half %acc_0_v2x0_sram_e1, i32 1
  %acc_0_c2d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_2, <64 x half> %acc_0_v2x0_sram, <64 x float> %acc_0_c1d0) #3
  %acc_0_v2x1_row = add i32 %morton_y, 16
  %acc_0_v2x1_col = add i32 %morton_x, 8
  %acc_0_v2x1_addr = mul i32 %acc_0_v2x1_row, 16
  %acc_0_v2x1_addr2 = add i32 %acc_0_v2x1_addr, %acc_0_v2x1_col
  %acc_0_v2x1_byte = mul i32 %acc_0_v2x1_addr2, 2
  %acc_0_v2x1_byte64 = zext i32 %acc_0_v2x1_byte to i64
  %acc_0_v2x1_byte64o = add i64 %acc_0_v2x1_byte64, 2048
  %acc_0_v2x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v2x1_byte64o
  %acc_0_v2x1_typed = bitcast i8 addrspace(3)* %acc_0_v2x1_ptr to <2 x half> addrspace(3)*
  %acc_0_v2x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v2x1_typed, align 4
  %acc_0_v2x1_v2 = bitcast <2 x half> %acc_0_v2x1_load to <2 x half>
  %acc_0_v2x1_sram_e0 = extractelement <2 x half> %acc_0_v2x1_v2, i32 0
  %acc_0_v2x1_sram_e1 = extractelement <2 x half> %acc_0_v2x1_v2, i32 1
  %acc_0_v2x1_sram_v0 = insertelement <64 x half> undef, half %acc_0_v2x1_sram_e0, i32 0
  %acc_0_v2x1_sram = insertelement <64 x half> %acc_0_v2x1_sram_v0, half %acc_0_v2x1_sram_e1, i32 1
  %acc_0_c2d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_2, <64 x half> %acc_0_v2x1_sram, <64 x float> %acc_0_c1d1) #3
  %acc_0_v3x0_row = add i32 %morton_y, 24
  %acc_0_v3x0_col = add i32 %morton_x, 0
  %acc_0_v3x0_addr = mul i32 %acc_0_v3x0_row, 16
  %acc_0_v3x0_addr2 = add i32 %acc_0_v3x0_addr, %acc_0_v3x0_col
  %acc_0_v3x0_byte = mul i32 %acc_0_v3x0_addr2, 2
  %acc_0_v3x0_byte64 = zext i32 %acc_0_v3x0_byte to i64
  %acc_0_v3x0_byte64o = add i64 %acc_0_v3x0_byte64, 2048
  %acc_0_v3x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v3x0_byte64o
  %acc_0_v3x0_typed = bitcast i8 addrspace(3)* %acc_0_v3x0_ptr to <2 x half> addrspace(3)*
  %acc_0_v3x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v3x0_typed, align 4
  %acc_0_v3x0_v2 = bitcast <2 x half> %acc_0_v3x0_load to <2 x half>
  %acc_0_v3x0_sram_e0 = extractelement <2 x half> %acc_0_v3x0_v2, i32 0
  %acc_0_v3x0_sram_e1 = extractelement <2 x half> %acc_0_v3x0_v2, i32 1
  %acc_0_v3x0_sram_v0 = insertelement <64 x half> undef, half %acc_0_v3x0_sram_e0, i32 0
  %acc_0_v3x0_sram = insertelement <64 x half> %acc_0_v3x0_sram_v0, half %acc_0_v3x0_sram_e1, i32 1
  %acc_0_c3d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_3, <64 x half> %acc_0_v3x0_sram, <64 x float> %acc_0_c2d0) #3
  %acc_0_v3x1_row = add i32 %morton_y, 24
  %acc_0_v3x1_col = add i32 %morton_x, 8
  %acc_0_v3x1_addr = mul i32 %acc_0_v3x1_row, 16
  %acc_0_v3x1_addr2 = add i32 %acc_0_v3x1_addr, %acc_0_v3x1_col
  %acc_0_v3x1_byte = mul i32 %acc_0_v3x1_addr2, 2
  %acc_0_v3x1_byte64 = zext i32 %acc_0_v3x1_byte to i64
  %acc_0_v3x1_byte64o = add i64 %acc_0_v3x1_byte64, 2048
  %acc_0_v3x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v3x1_byte64o
  %acc_0_v3x1_typed = bitcast i8 addrspace(3)* %acc_0_v3x1_ptr to <2 x half> addrspace(3)*
  %acc_0_v3x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v3x1_typed, align 4
  %acc_0_v3x1_v2 = bitcast <2 x half> %acc_0_v3x1_load to <2 x half>
  %acc_0_v3x1_sram_e0 = extractelement <2 x half> %acc_0_v3x1_v2, i32 0
  %acc_0_v3x1_sram_e1 = extractelement <2 x half> %acc_0_v3x1_v2, i32 1
  %acc_0_v3x1_sram_v0 = insertelement <64 x half> undef, half %acc_0_v3x1_sram_e0, i32 0
  %acc_0_v3x1_sram = insertelement <64 x half> %acc_0_v3x1_sram_v0, half %acc_0_v3x1_sram_e1, i32 1
  %acc_0_c3d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_3, <64 x half> %acc_0_v3x1_sram, <64 x float> %acc_0_c2d1) #3
  %acc_0_v4x0_row = add i32 %morton_y, 32
  %acc_0_v4x0_col = add i32 %morton_x, 0
  %acc_0_v4x0_addr = mul i32 %acc_0_v4x0_row, 16
  %acc_0_v4x0_addr2 = add i32 %acc_0_v4x0_addr, %acc_0_v4x0_col
  %acc_0_v4x0_byte = mul i32 %acc_0_v4x0_addr2, 2
  %acc_0_v4x0_byte64 = zext i32 %acc_0_v4x0_byte to i64
  %acc_0_v4x0_byte64o = add i64 %acc_0_v4x0_byte64, 2048
  %acc_0_v4x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v4x0_byte64o
  %acc_0_v4x0_typed = bitcast i8 addrspace(3)* %acc_0_v4x0_ptr to <2 x half> addrspace(3)*
  %acc_0_v4x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v4x0_typed, align 4
  %acc_0_v4x0_v2 = bitcast <2 x half> %acc_0_v4x0_load to <2 x half>
  %acc_0_v4x0_sram_e0 = extractelement <2 x half> %acc_0_v4x0_v2, i32 0
  %acc_0_v4x0_sram_e1 = extractelement <2 x half> %acc_0_v4x0_v2, i32 1
  %acc_0_v4x0_sram_v0 = insertelement <64 x half> undef, half %acc_0_v4x0_sram_e0, i32 0
  %acc_0_v4x0_sram = insertelement <64 x half> %acc_0_v4x0_sram_v0, half %acc_0_v4x0_sram_e1, i32 1
  %acc_0_c4d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_4, <64 x half> %acc_0_v4x0_sram, <64 x float> %acc_0_c3d0) #3
  %acc_0_v4x1_row = add i32 %morton_y, 32
  %acc_0_v4x1_col = add i32 %morton_x, 8
  %acc_0_v4x1_addr = mul i32 %acc_0_v4x1_row, 16
  %acc_0_v4x1_addr2 = add i32 %acc_0_v4x1_addr, %acc_0_v4x1_col
  %acc_0_v4x1_byte = mul i32 %acc_0_v4x1_addr2, 2
  %acc_0_v4x1_byte64 = zext i32 %acc_0_v4x1_byte to i64
  %acc_0_v4x1_byte64o = add i64 %acc_0_v4x1_byte64, 2048
  %acc_0_v4x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v4x1_byte64o
  %acc_0_v4x1_typed = bitcast i8 addrspace(3)* %acc_0_v4x1_ptr to <2 x half> addrspace(3)*
  %acc_0_v4x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v4x1_typed, align 4
  %acc_0_v4x1_v2 = bitcast <2 x half> %acc_0_v4x1_load to <2 x half>
  %acc_0_v4x1_sram_e0 = extractelement <2 x half> %acc_0_v4x1_v2, i32 0
  %acc_0_v4x1_sram_e1 = extractelement <2 x half> %acc_0_v4x1_v2, i32 1
  %acc_0_v4x1_sram_v0 = insertelement <64 x half> undef, half %acc_0_v4x1_sram_e0, i32 0
  %acc_0_v4x1_sram = insertelement <64 x half> %acc_0_v4x1_sram_v0, half %acc_0_v4x1_sram_e1, i32 1
  %acc_0_c4d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_4, <64 x half> %acc_0_v4x1_sram, <64 x float> %acc_0_c3d1) #3
  %acc_0_v5x0_row = add i32 %morton_y, 40
  %acc_0_v5x0_col = add i32 %morton_x, 0
  %acc_0_v5x0_addr = mul i32 %acc_0_v5x0_row, 16
  %acc_0_v5x0_addr2 = add i32 %acc_0_v5x0_addr, %acc_0_v5x0_col
  %acc_0_v5x0_byte = mul i32 %acc_0_v5x0_addr2, 2
  %acc_0_v5x0_byte64 = zext i32 %acc_0_v5x0_byte to i64
  %acc_0_v5x0_byte64o = add i64 %acc_0_v5x0_byte64, 2048
  %acc_0_v5x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v5x0_byte64o
  %acc_0_v5x0_typed = bitcast i8 addrspace(3)* %acc_0_v5x0_ptr to <2 x half> addrspace(3)*
  %acc_0_v5x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v5x0_typed, align 4
  %acc_0_v5x0_v2 = bitcast <2 x half> %acc_0_v5x0_load to <2 x half>
  %acc_0_v5x0_sram_e0 = extractelement <2 x half> %acc_0_v5x0_v2, i32 0
  %acc_0_v5x0_sram_e1 = extractelement <2 x half> %acc_0_v5x0_v2, i32 1
  %acc_0_v5x0_sram_v0 = insertelement <64 x half> undef, half %acc_0_v5x0_sram_e0, i32 0
  %acc_0_v5x0_sram = insertelement <64 x half> %acc_0_v5x0_sram_v0, half %acc_0_v5x0_sram_e1, i32 1
  %acc_0_c5d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_5, <64 x half> %acc_0_v5x0_sram, <64 x float> %acc_0_c4d0) #3
  %acc_0_v5x1_row = add i32 %morton_y, 40
  %acc_0_v5x1_col = add i32 %morton_x, 8
  %acc_0_v5x1_addr = mul i32 %acc_0_v5x1_row, 16
  %acc_0_v5x1_addr2 = add i32 %acc_0_v5x1_addr, %acc_0_v5x1_col
  %acc_0_v5x1_byte = mul i32 %acc_0_v5x1_addr2, 2
  %acc_0_v5x1_byte64 = zext i32 %acc_0_v5x1_byte to i64
  %acc_0_v5x1_byte64o = add i64 %acc_0_v5x1_byte64, 2048
  %acc_0_v5x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v5x1_byte64o
  %acc_0_v5x1_typed = bitcast i8 addrspace(3)* %acc_0_v5x1_ptr to <2 x half> addrspace(3)*
  %acc_0_v5x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v5x1_typed, align 4
  %acc_0_v5x1_v2 = bitcast <2 x half> %acc_0_v5x1_load to <2 x half>
  %acc_0_v5x1_sram_e0 = extractelement <2 x half> %acc_0_v5x1_v2, i32 0
  %acc_0_v5x1_sram_e1 = extractelement <2 x half> %acc_0_v5x1_v2, i32 1
  %acc_0_v5x1_sram_v0 = insertelement <64 x half> undef, half %acc_0_v5x1_sram_e0, i32 0
  %acc_0_v5x1_sram = insertelement <64 x half> %acc_0_v5x1_sram_v0, half %acc_0_v5x1_sram_e1, i32 1
  %acc_0_c5d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_5, <64 x half> %acc_0_v5x1_sram, <64 x float> %acc_0_c4d1) #3
  %acc_0_v6x0_row = add i32 %morton_y, 48
  %acc_0_v6x0_col = add i32 %morton_x, 0
  %acc_0_v6x0_addr = mul i32 %acc_0_v6x0_row, 16
  %acc_0_v6x0_addr2 = add i32 %acc_0_v6x0_addr, %acc_0_v6x0_col
  %acc_0_v6x0_byte = mul i32 %acc_0_v6x0_addr2, 2
  %acc_0_v6x0_byte64 = zext i32 %acc_0_v6x0_byte to i64
  %acc_0_v6x0_byte64o = add i64 %acc_0_v6x0_byte64, 2048
  %acc_0_v6x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v6x0_byte64o
  %acc_0_v6x0_typed = bitcast i8 addrspace(3)* %acc_0_v6x0_ptr to <2 x half> addrspace(3)*
  %acc_0_v6x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v6x0_typed, align 4
  %acc_0_v6x0_v2 = bitcast <2 x half> %acc_0_v6x0_load to <2 x half>
  %acc_0_v6x0_sram_e0 = extractelement <2 x half> %acc_0_v6x0_v2, i32 0
  %acc_0_v6x0_sram_e1 = extractelement <2 x half> %acc_0_v6x0_v2, i32 1
  %acc_0_v6x0_sram_v0 = insertelement <64 x half> undef, half %acc_0_v6x0_sram_e0, i32 0
  %acc_0_v6x0_sram = insertelement <64 x half> %acc_0_v6x0_sram_v0, half %acc_0_v6x0_sram_e1, i32 1
  %acc_0_c6d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_6, <64 x half> %acc_0_v6x0_sram, <64 x float> %acc_0_c5d0) #3
  %acc_0_v6x1_row = add i32 %morton_y, 48
  %acc_0_v6x1_col = add i32 %morton_x, 8
  %acc_0_v6x1_addr = mul i32 %acc_0_v6x1_row, 16
  %acc_0_v6x1_addr2 = add i32 %acc_0_v6x1_addr, %acc_0_v6x1_col
  %acc_0_v6x1_byte = mul i32 %acc_0_v6x1_addr2, 2
  %acc_0_v6x1_byte64 = zext i32 %acc_0_v6x1_byte to i64
  %acc_0_v6x1_byte64o = add i64 %acc_0_v6x1_byte64, 2048
  %acc_0_v6x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v6x1_byte64o
  %acc_0_v6x1_typed = bitcast i8 addrspace(3)* %acc_0_v6x1_ptr to <2 x half> addrspace(3)*
  %acc_0_v6x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v6x1_typed, align 4
  %acc_0_v6x1_v2 = bitcast <2 x half> %acc_0_v6x1_load to <2 x half>
  %acc_0_v6x1_sram_e0 = extractelement <2 x half> %acc_0_v6x1_v2, i32 0
  %acc_0_v6x1_sram_e1 = extractelement <2 x half> %acc_0_v6x1_v2, i32 1
  %acc_0_v6x1_sram_v0 = insertelement <64 x half> undef, half %acc_0_v6x1_sram_e0, i32 0
  %acc_0_v6x1_sram = insertelement <64 x half> %acc_0_v6x1_sram_v0, half %acc_0_v6x1_sram_e1, i32 1
  %acc_0_c6d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_6, <64 x half> %acc_0_v6x1_sram, <64 x float> %acc_0_c5d1) #3
  %acc_0_v7x0_row = add i32 %morton_y, 56
  %acc_0_v7x0_col = add i32 %morton_x, 0
  %acc_0_v7x0_addr = mul i32 %acc_0_v7x0_row, 16
  %acc_0_v7x0_addr2 = add i32 %acc_0_v7x0_addr, %acc_0_v7x0_col
  %acc_0_v7x0_byte = mul i32 %acc_0_v7x0_addr2, 2
  %acc_0_v7x0_byte64 = zext i32 %acc_0_v7x0_byte to i64
  %acc_0_v7x0_byte64o = add i64 %acc_0_v7x0_byte64, 2048
  %acc_0_v7x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v7x0_byte64o
  %acc_0_v7x0_typed = bitcast i8 addrspace(3)* %acc_0_v7x0_ptr to <2 x half> addrspace(3)*
  %acc_0_v7x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v7x0_typed, align 4
  %acc_0_v7x0_v2 = bitcast <2 x half> %acc_0_v7x0_load to <2 x half>
  %acc_0_v7x0_sram_e0 = extractelement <2 x half> %acc_0_v7x0_v2, i32 0
  %acc_0_v7x0_sram_e1 = extractelement <2 x half> %acc_0_v7x0_v2, i32 1
  %acc_0_v7x0_sram_v0 = insertelement <64 x half> undef, half %acc_0_v7x0_sram_e0, i32 0
  %acc_0_v7x0_sram = insertelement <64 x half> %acc_0_v7x0_sram_v0, half %acc_0_v7x0_sram_e1, i32 1
  %acc_0_c7d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_7, <64 x half> %acc_0_v7x0_sram, <64 x float> %acc_0_c6d0) #3
  %acc_0_v7x1_row = add i32 %morton_y, 56
  %acc_0_v7x1_col = add i32 %morton_x, 8
  %acc_0_v7x1_addr = mul i32 %acc_0_v7x1_row, 16
  %acc_0_v7x1_addr2 = add i32 %acc_0_v7x1_addr, %acc_0_v7x1_col
  %acc_0_v7x1_byte = mul i32 %acc_0_v7x1_addr2, 2
  %acc_0_v7x1_byte64 = zext i32 %acc_0_v7x1_byte to i64
  %acc_0_v7x1_byte64o = add i64 %acc_0_v7x1_byte64, 2048
  %acc_0_v7x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_0_v7x1_byte64o
  %acc_0_v7x1_typed = bitcast i8 addrspace(3)* %acc_0_v7x1_ptr to <2 x half> addrspace(3)*
  %acc_0_v7x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_0_v7x1_typed, align 4
  %acc_0_v7x1_v2 = bitcast <2 x half> %acc_0_v7x1_load to <2 x half>
  %acc_0_v7x1_sram_e0 = extractelement <2 x half> %acc_0_v7x1_v2, i32 0
  %acc_0_v7x1_sram_e1 = extractelement <2 x half> %acc_0_v7x1_v2, i32 1
  %acc_0_v7x1_sram_v0 = insertelement <64 x half> undef, half %acc_0_v7x1_sram_e0, i32 0
  %acc_0_v7x1_sram = insertelement <64 x half> %acc_0_v7x1_sram_v0, half %acc_0_v7x1_sram_e1, i32 1
  %acc_0_c7d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_7, <64 x half> %acc_0_v7x1_sram, <64 x float> %acc_0_c6d1) #3
  call void @air.wg.barrier(i32 2, i32 1)

  ; === Sync copy (acc_1_v) — all threads cooperative ===
  %acc_1_vsc_t0 = mul i32 %sidx, 32
  %acc_1_vsc_tid = add i32 %acc_1_vsc_t0, %lane_id
  %acc_1_vsc_drem = sub i32 64, 16
  %acc_1_vsc_dcmp = icmp ult i32 %acc_1_vsc_drem, 16
  %acc_1_vsc_dsrc = select i1 %acc_1_vsc_dcmp, i32 %acc_1_vsc_drem, i32 16
  %acc_1_vsc_srem = sub i32 64, %c
  %acc_1_vsc_scmp = icmp ult i32 %acc_1_vsc_srem, 64
  %acc_1_vsc_ssrc = select i1 %acc_1_vsc_scmp, i32 %acc_1_vsc_srem, i32 64
  br label %acc_1_vsc_pre

acc_1_vsc_pre:
  br label %acc_1_vsc_hdr

acc_1_vsc_hdr:
  %acc_1_vsc_i = phi i32 [%acc_1_vsc_tid, %acc_1_vsc_pre], [%acc_1_vsc_inx, %acc_1_vsc_st]
  %acc_1_vsc_done = icmp uge i32 %acc_1_vsc_i, 1024
  br i1 %acc_1_vsc_done, label %acc_1_vsc_end, label %acc_1_vsc_body

acc_1_vsc_body:
  %acc_1_vsc_row = lshr i32 %acc_1_vsc_i, 4
  %acc_1_vsc_col = and i32 %acc_1_vsc_i, 15
  %acc_1_vsc_rok = icmp ult i32 %acc_1_vsc_row, %acc_1_vsc_ssrc
  %acc_1_vsc_cok = icmp ult i32 %acc_1_vsc_col, %acc_1_vsc_dsrc
  %acc_1_vsc_ib = and i1 %acc_1_vsc_rok, %acc_1_vsc_cok
  br i1 %acc_1_vsc_ib, label %acc_1_vsc_ld, label %acc_1_vsc_zr

acc_1_vsc_ld:
  %acc_1_vsc_sr = add i32 %c, %acc_1_vsc_row
  %acc_1_vsc_sa = mul i32 %acc_1_vsc_sr, 64
  %acc_1_vsc_sc = add i32 16, %acc_1_vsc_col
  %acc_1_vsc_sad = add i32 %acc_1_vsc_sa, %acc_1_vsc_sc
  %acc_1_vsc_soff = zext i32 %acc_1_vsc_sad to i64
  %acc_1_vsc_sbyt = mul i64 %acc_1_vsc_soff, 2
  %acc_1_vsc_sp = getelementptr i8, i8 addrspace(1)* %V, i64 %acc_1_vsc_sbyt
  %acc_1_vsc_spt = bitcast i8 addrspace(1)* %acc_1_vsc_sp to i16 addrspace(1)*
  %acc_1_vsc_lv = load i16, i16 addrspace(1)* %acc_1_vsc_spt
  br label %acc_1_vsc_st

acc_1_vsc_zr:
  br label %acc_1_vsc_st

acc_1_vsc_st:
  %acc_1_vsc_val = phi i16 [%acc_1_vsc_lv, %acc_1_vsc_ld], [0, %acc_1_vsc_zr]
  %acc_1_vsc_tr = mul i32 %acc_1_vsc_row, 16
  %acc_1_vsc_ta = add i32 %acc_1_vsc_tr, %acc_1_vsc_col
  %acc_1_vsc_tb = mul i32 %acc_1_vsc_ta, 2
  %acc_1_vsc_tb64 = zext i32 %acc_1_vsc_tb to i64
  %acc_1_vsc_tb64o = add i64 %acc_1_vsc_tb64, 2048
  %acc_1_vsc_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_vsc_tb64o
  %acc_1_vsc_tpt = bitcast i8 addrspace(3)* %acc_1_vsc_tp to i16 addrspace(3)*
  store i16 %acc_1_vsc_val, i16 addrspace(3)* %acc_1_vsc_tpt
  %acc_1_vsc_inx = add i32 %acc_1_vsc_i, 64
  br label %acc_1_vsc_hdr

acc_1_vsc_end:
  call void @air.wg.barrier(i32 2, i32 1)

  %acc_1_v0x0_row = add i32 %morton_y, 0
  %acc_1_v0x0_col = add i32 %morton_x, 0
  %acc_1_v0x0_addr = mul i32 %acc_1_v0x0_row, 16
  %acc_1_v0x0_addr2 = add i32 %acc_1_v0x0_addr, %acc_1_v0x0_col
  %acc_1_v0x0_byte = mul i32 %acc_1_v0x0_addr2, 2
  %acc_1_v0x0_byte64 = zext i32 %acc_1_v0x0_byte to i64
  %acc_1_v0x0_byte64o = add i64 %acc_1_v0x0_byte64, 2048
  %acc_1_v0x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v0x0_byte64o
  %acc_1_v0x0_typed = bitcast i8 addrspace(3)* %acc_1_v0x0_ptr to <2 x half> addrspace(3)*
  %acc_1_v0x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v0x0_typed, align 4
  %acc_1_v0x0_v2 = bitcast <2 x half> %acc_1_v0x0_load to <2 x half>
  %acc_1_v0x0_sram_e0 = extractelement <2 x half> %acc_1_v0x0_v2, i32 0
  %acc_1_v0x0_sram_e1 = extractelement <2 x half> %acc_1_v0x0_v2, i32 1
  %acc_1_v0x0_sram_v0 = insertelement <64 x half> undef, half %acc_1_v0x0_sram_e0, i32 0
  %acc_1_v0x0_sram = insertelement <64 x half> %acc_1_v0x0_sram_v0, half %acc_1_v0x0_sram_e1, i32 1
  %acc_1_c0d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_0, <64 x half> %acc_1_v0x0_sram, <64 x float> %acc_corrected_2) #3
  %acc_1_v0x1_row = add i32 %morton_y, 0
  %acc_1_v0x1_col = add i32 %morton_x, 8
  %acc_1_v0x1_addr = mul i32 %acc_1_v0x1_row, 16
  %acc_1_v0x1_addr2 = add i32 %acc_1_v0x1_addr, %acc_1_v0x1_col
  %acc_1_v0x1_byte = mul i32 %acc_1_v0x1_addr2, 2
  %acc_1_v0x1_byte64 = zext i32 %acc_1_v0x1_byte to i64
  %acc_1_v0x1_byte64o = add i64 %acc_1_v0x1_byte64, 2048
  %acc_1_v0x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v0x1_byte64o
  %acc_1_v0x1_typed = bitcast i8 addrspace(3)* %acc_1_v0x1_ptr to <2 x half> addrspace(3)*
  %acc_1_v0x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v0x1_typed, align 4
  %acc_1_v0x1_v2 = bitcast <2 x half> %acc_1_v0x1_load to <2 x half>
  %acc_1_v0x1_sram_e0 = extractelement <2 x half> %acc_1_v0x1_v2, i32 0
  %acc_1_v0x1_sram_e1 = extractelement <2 x half> %acc_1_v0x1_v2, i32 1
  %acc_1_v0x1_sram_v0 = insertelement <64 x half> undef, half %acc_1_v0x1_sram_e0, i32 0
  %acc_1_v0x1_sram = insertelement <64 x half> %acc_1_v0x1_sram_v0, half %acc_1_v0x1_sram_e1, i32 1
  %acc_1_c0d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_0, <64 x half> %acc_1_v0x1_sram, <64 x float> %acc_corrected_3) #3
  %acc_1_v1x0_row = add i32 %morton_y, 8
  %acc_1_v1x0_col = add i32 %morton_x, 0
  %acc_1_v1x0_addr = mul i32 %acc_1_v1x0_row, 16
  %acc_1_v1x0_addr2 = add i32 %acc_1_v1x0_addr, %acc_1_v1x0_col
  %acc_1_v1x0_byte = mul i32 %acc_1_v1x0_addr2, 2
  %acc_1_v1x0_byte64 = zext i32 %acc_1_v1x0_byte to i64
  %acc_1_v1x0_byte64o = add i64 %acc_1_v1x0_byte64, 2048
  %acc_1_v1x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v1x0_byte64o
  %acc_1_v1x0_typed = bitcast i8 addrspace(3)* %acc_1_v1x0_ptr to <2 x half> addrspace(3)*
  %acc_1_v1x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v1x0_typed, align 4
  %acc_1_v1x0_v2 = bitcast <2 x half> %acc_1_v1x0_load to <2 x half>
  %acc_1_v1x0_sram_e0 = extractelement <2 x half> %acc_1_v1x0_v2, i32 0
  %acc_1_v1x0_sram_e1 = extractelement <2 x half> %acc_1_v1x0_v2, i32 1
  %acc_1_v1x0_sram_v0 = insertelement <64 x half> undef, half %acc_1_v1x0_sram_e0, i32 0
  %acc_1_v1x0_sram = insertelement <64 x half> %acc_1_v1x0_sram_v0, half %acc_1_v1x0_sram_e1, i32 1
  %acc_1_c1d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_1, <64 x half> %acc_1_v1x0_sram, <64 x float> %acc_1_c0d0) #3
  %acc_1_v1x1_row = add i32 %morton_y, 8
  %acc_1_v1x1_col = add i32 %morton_x, 8
  %acc_1_v1x1_addr = mul i32 %acc_1_v1x1_row, 16
  %acc_1_v1x1_addr2 = add i32 %acc_1_v1x1_addr, %acc_1_v1x1_col
  %acc_1_v1x1_byte = mul i32 %acc_1_v1x1_addr2, 2
  %acc_1_v1x1_byte64 = zext i32 %acc_1_v1x1_byte to i64
  %acc_1_v1x1_byte64o = add i64 %acc_1_v1x1_byte64, 2048
  %acc_1_v1x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v1x1_byte64o
  %acc_1_v1x1_typed = bitcast i8 addrspace(3)* %acc_1_v1x1_ptr to <2 x half> addrspace(3)*
  %acc_1_v1x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v1x1_typed, align 4
  %acc_1_v1x1_v2 = bitcast <2 x half> %acc_1_v1x1_load to <2 x half>
  %acc_1_v1x1_sram_e0 = extractelement <2 x half> %acc_1_v1x1_v2, i32 0
  %acc_1_v1x1_sram_e1 = extractelement <2 x half> %acc_1_v1x1_v2, i32 1
  %acc_1_v1x1_sram_v0 = insertelement <64 x half> undef, half %acc_1_v1x1_sram_e0, i32 0
  %acc_1_v1x1_sram = insertelement <64 x half> %acc_1_v1x1_sram_v0, half %acc_1_v1x1_sram_e1, i32 1
  %acc_1_c1d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_1, <64 x half> %acc_1_v1x1_sram, <64 x float> %acc_1_c0d1) #3
  %acc_1_v2x0_row = add i32 %morton_y, 16
  %acc_1_v2x0_col = add i32 %morton_x, 0
  %acc_1_v2x0_addr = mul i32 %acc_1_v2x0_row, 16
  %acc_1_v2x0_addr2 = add i32 %acc_1_v2x0_addr, %acc_1_v2x0_col
  %acc_1_v2x0_byte = mul i32 %acc_1_v2x0_addr2, 2
  %acc_1_v2x0_byte64 = zext i32 %acc_1_v2x0_byte to i64
  %acc_1_v2x0_byte64o = add i64 %acc_1_v2x0_byte64, 2048
  %acc_1_v2x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v2x0_byte64o
  %acc_1_v2x0_typed = bitcast i8 addrspace(3)* %acc_1_v2x0_ptr to <2 x half> addrspace(3)*
  %acc_1_v2x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v2x0_typed, align 4
  %acc_1_v2x0_v2 = bitcast <2 x half> %acc_1_v2x0_load to <2 x half>
  %acc_1_v2x0_sram_e0 = extractelement <2 x half> %acc_1_v2x0_v2, i32 0
  %acc_1_v2x0_sram_e1 = extractelement <2 x half> %acc_1_v2x0_v2, i32 1
  %acc_1_v2x0_sram_v0 = insertelement <64 x half> undef, half %acc_1_v2x0_sram_e0, i32 0
  %acc_1_v2x0_sram = insertelement <64 x half> %acc_1_v2x0_sram_v0, half %acc_1_v2x0_sram_e1, i32 1
  %acc_1_c2d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_2, <64 x half> %acc_1_v2x0_sram, <64 x float> %acc_1_c1d0) #3
  %acc_1_v2x1_row = add i32 %morton_y, 16
  %acc_1_v2x1_col = add i32 %morton_x, 8
  %acc_1_v2x1_addr = mul i32 %acc_1_v2x1_row, 16
  %acc_1_v2x1_addr2 = add i32 %acc_1_v2x1_addr, %acc_1_v2x1_col
  %acc_1_v2x1_byte = mul i32 %acc_1_v2x1_addr2, 2
  %acc_1_v2x1_byte64 = zext i32 %acc_1_v2x1_byte to i64
  %acc_1_v2x1_byte64o = add i64 %acc_1_v2x1_byte64, 2048
  %acc_1_v2x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v2x1_byte64o
  %acc_1_v2x1_typed = bitcast i8 addrspace(3)* %acc_1_v2x1_ptr to <2 x half> addrspace(3)*
  %acc_1_v2x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v2x1_typed, align 4
  %acc_1_v2x1_v2 = bitcast <2 x half> %acc_1_v2x1_load to <2 x half>
  %acc_1_v2x1_sram_e0 = extractelement <2 x half> %acc_1_v2x1_v2, i32 0
  %acc_1_v2x1_sram_e1 = extractelement <2 x half> %acc_1_v2x1_v2, i32 1
  %acc_1_v2x1_sram_v0 = insertelement <64 x half> undef, half %acc_1_v2x1_sram_e0, i32 0
  %acc_1_v2x1_sram = insertelement <64 x half> %acc_1_v2x1_sram_v0, half %acc_1_v2x1_sram_e1, i32 1
  %acc_1_c2d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_2, <64 x half> %acc_1_v2x1_sram, <64 x float> %acc_1_c1d1) #3
  %acc_1_v3x0_row = add i32 %morton_y, 24
  %acc_1_v3x0_col = add i32 %morton_x, 0
  %acc_1_v3x0_addr = mul i32 %acc_1_v3x0_row, 16
  %acc_1_v3x0_addr2 = add i32 %acc_1_v3x0_addr, %acc_1_v3x0_col
  %acc_1_v3x0_byte = mul i32 %acc_1_v3x0_addr2, 2
  %acc_1_v3x0_byte64 = zext i32 %acc_1_v3x0_byte to i64
  %acc_1_v3x0_byte64o = add i64 %acc_1_v3x0_byte64, 2048
  %acc_1_v3x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v3x0_byte64o
  %acc_1_v3x0_typed = bitcast i8 addrspace(3)* %acc_1_v3x0_ptr to <2 x half> addrspace(3)*
  %acc_1_v3x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v3x0_typed, align 4
  %acc_1_v3x0_v2 = bitcast <2 x half> %acc_1_v3x0_load to <2 x half>
  %acc_1_v3x0_sram_e0 = extractelement <2 x half> %acc_1_v3x0_v2, i32 0
  %acc_1_v3x0_sram_e1 = extractelement <2 x half> %acc_1_v3x0_v2, i32 1
  %acc_1_v3x0_sram_v0 = insertelement <64 x half> undef, half %acc_1_v3x0_sram_e0, i32 0
  %acc_1_v3x0_sram = insertelement <64 x half> %acc_1_v3x0_sram_v0, half %acc_1_v3x0_sram_e1, i32 1
  %acc_1_c3d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_3, <64 x half> %acc_1_v3x0_sram, <64 x float> %acc_1_c2d0) #3
  %acc_1_v3x1_row = add i32 %morton_y, 24
  %acc_1_v3x1_col = add i32 %morton_x, 8
  %acc_1_v3x1_addr = mul i32 %acc_1_v3x1_row, 16
  %acc_1_v3x1_addr2 = add i32 %acc_1_v3x1_addr, %acc_1_v3x1_col
  %acc_1_v3x1_byte = mul i32 %acc_1_v3x1_addr2, 2
  %acc_1_v3x1_byte64 = zext i32 %acc_1_v3x1_byte to i64
  %acc_1_v3x1_byte64o = add i64 %acc_1_v3x1_byte64, 2048
  %acc_1_v3x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v3x1_byte64o
  %acc_1_v3x1_typed = bitcast i8 addrspace(3)* %acc_1_v3x1_ptr to <2 x half> addrspace(3)*
  %acc_1_v3x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v3x1_typed, align 4
  %acc_1_v3x1_v2 = bitcast <2 x half> %acc_1_v3x1_load to <2 x half>
  %acc_1_v3x1_sram_e0 = extractelement <2 x half> %acc_1_v3x1_v2, i32 0
  %acc_1_v3x1_sram_e1 = extractelement <2 x half> %acc_1_v3x1_v2, i32 1
  %acc_1_v3x1_sram_v0 = insertelement <64 x half> undef, half %acc_1_v3x1_sram_e0, i32 0
  %acc_1_v3x1_sram = insertelement <64 x half> %acc_1_v3x1_sram_v0, half %acc_1_v3x1_sram_e1, i32 1
  %acc_1_c3d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_3, <64 x half> %acc_1_v3x1_sram, <64 x float> %acc_1_c2d1) #3
  %acc_1_v4x0_row = add i32 %morton_y, 32
  %acc_1_v4x0_col = add i32 %morton_x, 0
  %acc_1_v4x0_addr = mul i32 %acc_1_v4x0_row, 16
  %acc_1_v4x0_addr2 = add i32 %acc_1_v4x0_addr, %acc_1_v4x0_col
  %acc_1_v4x0_byte = mul i32 %acc_1_v4x0_addr2, 2
  %acc_1_v4x0_byte64 = zext i32 %acc_1_v4x0_byte to i64
  %acc_1_v4x0_byte64o = add i64 %acc_1_v4x0_byte64, 2048
  %acc_1_v4x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v4x0_byte64o
  %acc_1_v4x0_typed = bitcast i8 addrspace(3)* %acc_1_v4x0_ptr to <2 x half> addrspace(3)*
  %acc_1_v4x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v4x0_typed, align 4
  %acc_1_v4x0_v2 = bitcast <2 x half> %acc_1_v4x0_load to <2 x half>
  %acc_1_v4x0_sram_e0 = extractelement <2 x half> %acc_1_v4x0_v2, i32 0
  %acc_1_v4x0_sram_e1 = extractelement <2 x half> %acc_1_v4x0_v2, i32 1
  %acc_1_v4x0_sram_v0 = insertelement <64 x half> undef, half %acc_1_v4x0_sram_e0, i32 0
  %acc_1_v4x0_sram = insertelement <64 x half> %acc_1_v4x0_sram_v0, half %acc_1_v4x0_sram_e1, i32 1
  %acc_1_c4d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_4, <64 x half> %acc_1_v4x0_sram, <64 x float> %acc_1_c3d0) #3
  %acc_1_v4x1_row = add i32 %morton_y, 32
  %acc_1_v4x1_col = add i32 %morton_x, 8
  %acc_1_v4x1_addr = mul i32 %acc_1_v4x1_row, 16
  %acc_1_v4x1_addr2 = add i32 %acc_1_v4x1_addr, %acc_1_v4x1_col
  %acc_1_v4x1_byte = mul i32 %acc_1_v4x1_addr2, 2
  %acc_1_v4x1_byte64 = zext i32 %acc_1_v4x1_byte to i64
  %acc_1_v4x1_byte64o = add i64 %acc_1_v4x1_byte64, 2048
  %acc_1_v4x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v4x1_byte64o
  %acc_1_v4x1_typed = bitcast i8 addrspace(3)* %acc_1_v4x1_ptr to <2 x half> addrspace(3)*
  %acc_1_v4x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v4x1_typed, align 4
  %acc_1_v4x1_v2 = bitcast <2 x half> %acc_1_v4x1_load to <2 x half>
  %acc_1_v4x1_sram_e0 = extractelement <2 x half> %acc_1_v4x1_v2, i32 0
  %acc_1_v4x1_sram_e1 = extractelement <2 x half> %acc_1_v4x1_v2, i32 1
  %acc_1_v4x1_sram_v0 = insertelement <64 x half> undef, half %acc_1_v4x1_sram_e0, i32 0
  %acc_1_v4x1_sram = insertelement <64 x half> %acc_1_v4x1_sram_v0, half %acc_1_v4x1_sram_e1, i32 1
  %acc_1_c4d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_4, <64 x half> %acc_1_v4x1_sram, <64 x float> %acc_1_c3d1) #3
  %acc_1_v5x0_row = add i32 %morton_y, 40
  %acc_1_v5x0_col = add i32 %morton_x, 0
  %acc_1_v5x0_addr = mul i32 %acc_1_v5x0_row, 16
  %acc_1_v5x0_addr2 = add i32 %acc_1_v5x0_addr, %acc_1_v5x0_col
  %acc_1_v5x0_byte = mul i32 %acc_1_v5x0_addr2, 2
  %acc_1_v5x0_byte64 = zext i32 %acc_1_v5x0_byte to i64
  %acc_1_v5x0_byte64o = add i64 %acc_1_v5x0_byte64, 2048
  %acc_1_v5x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v5x0_byte64o
  %acc_1_v5x0_typed = bitcast i8 addrspace(3)* %acc_1_v5x0_ptr to <2 x half> addrspace(3)*
  %acc_1_v5x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v5x0_typed, align 4
  %acc_1_v5x0_v2 = bitcast <2 x half> %acc_1_v5x0_load to <2 x half>
  %acc_1_v5x0_sram_e0 = extractelement <2 x half> %acc_1_v5x0_v2, i32 0
  %acc_1_v5x0_sram_e1 = extractelement <2 x half> %acc_1_v5x0_v2, i32 1
  %acc_1_v5x0_sram_v0 = insertelement <64 x half> undef, half %acc_1_v5x0_sram_e0, i32 0
  %acc_1_v5x0_sram = insertelement <64 x half> %acc_1_v5x0_sram_v0, half %acc_1_v5x0_sram_e1, i32 1
  %acc_1_c5d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_5, <64 x half> %acc_1_v5x0_sram, <64 x float> %acc_1_c4d0) #3
  %acc_1_v5x1_row = add i32 %morton_y, 40
  %acc_1_v5x1_col = add i32 %morton_x, 8
  %acc_1_v5x1_addr = mul i32 %acc_1_v5x1_row, 16
  %acc_1_v5x1_addr2 = add i32 %acc_1_v5x1_addr, %acc_1_v5x1_col
  %acc_1_v5x1_byte = mul i32 %acc_1_v5x1_addr2, 2
  %acc_1_v5x1_byte64 = zext i32 %acc_1_v5x1_byte to i64
  %acc_1_v5x1_byte64o = add i64 %acc_1_v5x1_byte64, 2048
  %acc_1_v5x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v5x1_byte64o
  %acc_1_v5x1_typed = bitcast i8 addrspace(3)* %acc_1_v5x1_ptr to <2 x half> addrspace(3)*
  %acc_1_v5x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v5x1_typed, align 4
  %acc_1_v5x1_v2 = bitcast <2 x half> %acc_1_v5x1_load to <2 x half>
  %acc_1_v5x1_sram_e0 = extractelement <2 x half> %acc_1_v5x1_v2, i32 0
  %acc_1_v5x1_sram_e1 = extractelement <2 x half> %acc_1_v5x1_v2, i32 1
  %acc_1_v5x1_sram_v0 = insertelement <64 x half> undef, half %acc_1_v5x1_sram_e0, i32 0
  %acc_1_v5x1_sram = insertelement <64 x half> %acc_1_v5x1_sram_v0, half %acc_1_v5x1_sram_e1, i32 1
  %acc_1_c5d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_5, <64 x half> %acc_1_v5x1_sram, <64 x float> %acc_1_c4d1) #3
  %acc_1_v6x0_row = add i32 %morton_y, 48
  %acc_1_v6x0_col = add i32 %morton_x, 0
  %acc_1_v6x0_addr = mul i32 %acc_1_v6x0_row, 16
  %acc_1_v6x0_addr2 = add i32 %acc_1_v6x0_addr, %acc_1_v6x0_col
  %acc_1_v6x0_byte = mul i32 %acc_1_v6x0_addr2, 2
  %acc_1_v6x0_byte64 = zext i32 %acc_1_v6x0_byte to i64
  %acc_1_v6x0_byte64o = add i64 %acc_1_v6x0_byte64, 2048
  %acc_1_v6x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v6x0_byte64o
  %acc_1_v6x0_typed = bitcast i8 addrspace(3)* %acc_1_v6x0_ptr to <2 x half> addrspace(3)*
  %acc_1_v6x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v6x0_typed, align 4
  %acc_1_v6x0_v2 = bitcast <2 x half> %acc_1_v6x0_load to <2 x half>
  %acc_1_v6x0_sram_e0 = extractelement <2 x half> %acc_1_v6x0_v2, i32 0
  %acc_1_v6x0_sram_e1 = extractelement <2 x half> %acc_1_v6x0_v2, i32 1
  %acc_1_v6x0_sram_v0 = insertelement <64 x half> undef, half %acc_1_v6x0_sram_e0, i32 0
  %acc_1_v6x0_sram = insertelement <64 x half> %acc_1_v6x0_sram_v0, half %acc_1_v6x0_sram_e1, i32 1
  %acc_1_c6d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_6, <64 x half> %acc_1_v6x0_sram, <64 x float> %acc_1_c5d0) #3
  %acc_1_v6x1_row = add i32 %morton_y, 48
  %acc_1_v6x1_col = add i32 %morton_x, 8
  %acc_1_v6x1_addr = mul i32 %acc_1_v6x1_row, 16
  %acc_1_v6x1_addr2 = add i32 %acc_1_v6x1_addr, %acc_1_v6x1_col
  %acc_1_v6x1_byte = mul i32 %acc_1_v6x1_addr2, 2
  %acc_1_v6x1_byte64 = zext i32 %acc_1_v6x1_byte to i64
  %acc_1_v6x1_byte64o = add i64 %acc_1_v6x1_byte64, 2048
  %acc_1_v6x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v6x1_byte64o
  %acc_1_v6x1_typed = bitcast i8 addrspace(3)* %acc_1_v6x1_ptr to <2 x half> addrspace(3)*
  %acc_1_v6x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v6x1_typed, align 4
  %acc_1_v6x1_v2 = bitcast <2 x half> %acc_1_v6x1_load to <2 x half>
  %acc_1_v6x1_sram_e0 = extractelement <2 x half> %acc_1_v6x1_v2, i32 0
  %acc_1_v6x1_sram_e1 = extractelement <2 x half> %acc_1_v6x1_v2, i32 1
  %acc_1_v6x1_sram_v0 = insertelement <64 x half> undef, half %acc_1_v6x1_sram_e0, i32 0
  %acc_1_v6x1_sram = insertelement <64 x half> %acc_1_v6x1_sram_v0, half %acc_1_v6x1_sram_e1, i32 1
  %acc_1_c6d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_6, <64 x half> %acc_1_v6x1_sram, <64 x float> %acc_1_c5d1) #3
  %acc_1_v7x0_row = add i32 %morton_y, 56
  %acc_1_v7x0_col = add i32 %morton_x, 0
  %acc_1_v7x0_addr = mul i32 %acc_1_v7x0_row, 16
  %acc_1_v7x0_addr2 = add i32 %acc_1_v7x0_addr, %acc_1_v7x0_col
  %acc_1_v7x0_byte = mul i32 %acc_1_v7x0_addr2, 2
  %acc_1_v7x0_byte64 = zext i32 %acc_1_v7x0_byte to i64
  %acc_1_v7x0_byte64o = add i64 %acc_1_v7x0_byte64, 2048
  %acc_1_v7x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v7x0_byte64o
  %acc_1_v7x0_typed = bitcast i8 addrspace(3)* %acc_1_v7x0_ptr to <2 x half> addrspace(3)*
  %acc_1_v7x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v7x0_typed, align 4
  %acc_1_v7x0_v2 = bitcast <2 x half> %acc_1_v7x0_load to <2 x half>
  %acc_1_v7x0_sram_e0 = extractelement <2 x half> %acc_1_v7x0_v2, i32 0
  %acc_1_v7x0_sram_e1 = extractelement <2 x half> %acc_1_v7x0_v2, i32 1
  %acc_1_v7x0_sram_v0 = insertelement <64 x half> undef, half %acc_1_v7x0_sram_e0, i32 0
  %acc_1_v7x0_sram = insertelement <64 x half> %acc_1_v7x0_sram_v0, half %acc_1_v7x0_sram_e1, i32 1
  %acc_1_c7d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_7, <64 x half> %acc_1_v7x0_sram, <64 x float> %acc_1_c6d0) #3
  %acc_1_v7x1_row = add i32 %morton_y, 56
  %acc_1_v7x1_col = add i32 %morton_x, 8
  %acc_1_v7x1_addr = mul i32 %acc_1_v7x1_row, 16
  %acc_1_v7x1_addr2 = add i32 %acc_1_v7x1_addr, %acc_1_v7x1_col
  %acc_1_v7x1_byte = mul i32 %acc_1_v7x1_addr2, 2
  %acc_1_v7x1_byte64 = zext i32 %acc_1_v7x1_byte to i64
  %acc_1_v7x1_byte64o = add i64 %acc_1_v7x1_byte64, 2048
  %acc_1_v7x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_1_v7x1_byte64o
  %acc_1_v7x1_typed = bitcast i8 addrspace(3)* %acc_1_v7x1_ptr to <2 x half> addrspace(3)*
  %acc_1_v7x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_1_v7x1_typed, align 4
  %acc_1_v7x1_v2 = bitcast <2 x half> %acc_1_v7x1_load to <2 x half>
  %acc_1_v7x1_sram_e0 = extractelement <2 x half> %acc_1_v7x1_v2, i32 0
  %acc_1_v7x1_sram_e1 = extractelement <2 x half> %acc_1_v7x1_v2, i32 1
  %acc_1_v7x1_sram_v0 = insertelement <64 x half> undef, half %acc_1_v7x1_sram_e0, i32 0
  %acc_1_v7x1_sram = insertelement <64 x half> %acc_1_v7x1_sram_v0, half %acc_1_v7x1_sram_e1, i32 1
  %acc_1_c7d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_7, <64 x half> %acc_1_v7x1_sram, <64 x float> %acc_1_c6d1) #3
  call void @air.wg.barrier(i32 2, i32 1)

  ; === Sync copy (acc_2_v) — all threads cooperative ===
  %acc_2_vsc_t0 = mul i32 %sidx, 32
  %acc_2_vsc_tid = add i32 %acc_2_vsc_t0, %lane_id
  %acc_2_vsc_drem = sub i32 64, 32
  %acc_2_vsc_dcmp = icmp ult i32 %acc_2_vsc_drem, 16
  %acc_2_vsc_dsrc = select i1 %acc_2_vsc_dcmp, i32 %acc_2_vsc_drem, i32 16
  %acc_2_vsc_srem = sub i32 64, %c
  %acc_2_vsc_scmp = icmp ult i32 %acc_2_vsc_srem, 64
  %acc_2_vsc_ssrc = select i1 %acc_2_vsc_scmp, i32 %acc_2_vsc_srem, i32 64
  br label %acc_2_vsc_pre

acc_2_vsc_pre:
  br label %acc_2_vsc_hdr

acc_2_vsc_hdr:
  %acc_2_vsc_i = phi i32 [%acc_2_vsc_tid, %acc_2_vsc_pre], [%acc_2_vsc_inx, %acc_2_vsc_st]
  %acc_2_vsc_done = icmp uge i32 %acc_2_vsc_i, 1024
  br i1 %acc_2_vsc_done, label %acc_2_vsc_end, label %acc_2_vsc_body

acc_2_vsc_body:
  %acc_2_vsc_row = lshr i32 %acc_2_vsc_i, 4
  %acc_2_vsc_col = and i32 %acc_2_vsc_i, 15
  %acc_2_vsc_rok = icmp ult i32 %acc_2_vsc_row, %acc_2_vsc_ssrc
  %acc_2_vsc_cok = icmp ult i32 %acc_2_vsc_col, %acc_2_vsc_dsrc
  %acc_2_vsc_ib = and i1 %acc_2_vsc_rok, %acc_2_vsc_cok
  br i1 %acc_2_vsc_ib, label %acc_2_vsc_ld, label %acc_2_vsc_zr

acc_2_vsc_ld:
  %acc_2_vsc_sr = add i32 %c, %acc_2_vsc_row
  %acc_2_vsc_sa = mul i32 %acc_2_vsc_sr, 64
  %acc_2_vsc_sc = add i32 32, %acc_2_vsc_col
  %acc_2_vsc_sad = add i32 %acc_2_vsc_sa, %acc_2_vsc_sc
  %acc_2_vsc_soff = zext i32 %acc_2_vsc_sad to i64
  %acc_2_vsc_sbyt = mul i64 %acc_2_vsc_soff, 2
  %acc_2_vsc_sp = getelementptr i8, i8 addrspace(1)* %V, i64 %acc_2_vsc_sbyt
  %acc_2_vsc_spt = bitcast i8 addrspace(1)* %acc_2_vsc_sp to i16 addrspace(1)*
  %acc_2_vsc_lv = load i16, i16 addrspace(1)* %acc_2_vsc_spt
  br label %acc_2_vsc_st

acc_2_vsc_zr:
  br label %acc_2_vsc_st

acc_2_vsc_st:
  %acc_2_vsc_val = phi i16 [%acc_2_vsc_lv, %acc_2_vsc_ld], [0, %acc_2_vsc_zr]
  %acc_2_vsc_tr = mul i32 %acc_2_vsc_row, 16
  %acc_2_vsc_ta = add i32 %acc_2_vsc_tr, %acc_2_vsc_col
  %acc_2_vsc_tb = mul i32 %acc_2_vsc_ta, 2
  %acc_2_vsc_tb64 = zext i32 %acc_2_vsc_tb to i64
  %acc_2_vsc_tb64o = add i64 %acc_2_vsc_tb64, 2048
  %acc_2_vsc_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_vsc_tb64o
  %acc_2_vsc_tpt = bitcast i8 addrspace(3)* %acc_2_vsc_tp to i16 addrspace(3)*
  store i16 %acc_2_vsc_val, i16 addrspace(3)* %acc_2_vsc_tpt
  %acc_2_vsc_inx = add i32 %acc_2_vsc_i, 64
  br label %acc_2_vsc_hdr

acc_2_vsc_end:
  call void @air.wg.barrier(i32 2, i32 1)

  %acc_2_v0x0_row = add i32 %morton_y, 0
  %acc_2_v0x0_col = add i32 %morton_x, 0
  %acc_2_v0x0_addr = mul i32 %acc_2_v0x0_row, 16
  %acc_2_v0x0_addr2 = add i32 %acc_2_v0x0_addr, %acc_2_v0x0_col
  %acc_2_v0x0_byte = mul i32 %acc_2_v0x0_addr2, 2
  %acc_2_v0x0_byte64 = zext i32 %acc_2_v0x0_byte to i64
  %acc_2_v0x0_byte64o = add i64 %acc_2_v0x0_byte64, 2048
  %acc_2_v0x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v0x0_byte64o
  %acc_2_v0x0_typed = bitcast i8 addrspace(3)* %acc_2_v0x0_ptr to <2 x half> addrspace(3)*
  %acc_2_v0x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v0x0_typed, align 4
  %acc_2_v0x0_v2 = bitcast <2 x half> %acc_2_v0x0_load to <2 x half>
  %acc_2_v0x0_sram_e0 = extractelement <2 x half> %acc_2_v0x0_v2, i32 0
  %acc_2_v0x0_sram_e1 = extractelement <2 x half> %acc_2_v0x0_v2, i32 1
  %acc_2_v0x0_sram_v0 = insertelement <64 x half> undef, half %acc_2_v0x0_sram_e0, i32 0
  %acc_2_v0x0_sram = insertelement <64 x half> %acc_2_v0x0_sram_v0, half %acc_2_v0x0_sram_e1, i32 1
  %acc_2_c0d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_0, <64 x half> %acc_2_v0x0_sram, <64 x float> %acc_corrected_4) #3
  %acc_2_v0x1_row = add i32 %morton_y, 0
  %acc_2_v0x1_col = add i32 %morton_x, 8
  %acc_2_v0x1_addr = mul i32 %acc_2_v0x1_row, 16
  %acc_2_v0x1_addr2 = add i32 %acc_2_v0x1_addr, %acc_2_v0x1_col
  %acc_2_v0x1_byte = mul i32 %acc_2_v0x1_addr2, 2
  %acc_2_v0x1_byte64 = zext i32 %acc_2_v0x1_byte to i64
  %acc_2_v0x1_byte64o = add i64 %acc_2_v0x1_byte64, 2048
  %acc_2_v0x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v0x1_byte64o
  %acc_2_v0x1_typed = bitcast i8 addrspace(3)* %acc_2_v0x1_ptr to <2 x half> addrspace(3)*
  %acc_2_v0x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v0x1_typed, align 4
  %acc_2_v0x1_v2 = bitcast <2 x half> %acc_2_v0x1_load to <2 x half>
  %acc_2_v0x1_sram_e0 = extractelement <2 x half> %acc_2_v0x1_v2, i32 0
  %acc_2_v0x1_sram_e1 = extractelement <2 x half> %acc_2_v0x1_v2, i32 1
  %acc_2_v0x1_sram_v0 = insertelement <64 x half> undef, half %acc_2_v0x1_sram_e0, i32 0
  %acc_2_v0x1_sram = insertelement <64 x half> %acc_2_v0x1_sram_v0, half %acc_2_v0x1_sram_e1, i32 1
  %acc_2_c0d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_0, <64 x half> %acc_2_v0x1_sram, <64 x float> %acc_corrected_5) #3
  %acc_2_v1x0_row = add i32 %morton_y, 8
  %acc_2_v1x0_col = add i32 %morton_x, 0
  %acc_2_v1x0_addr = mul i32 %acc_2_v1x0_row, 16
  %acc_2_v1x0_addr2 = add i32 %acc_2_v1x0_addr, %acc_2_v1x0_col
  %acc_2_v1x0_byte = mul i32 %acc_2_v1x0_addr2, 2
  %acc_2_v1x0_byte64 = zext i32 %acc_2_v1x0_byte to i64
  %acc_2_v1x0_byte64o = add i64 %acc_2_v1x0_byte64, 2048
  %acc_2_v1x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v1x0_byte64o
  %acc_2_v1x0_typed = bitcast i8 addrspace(3)* %acc_2_v1x0_ptr to <2 x half> addrspace(3)*
  %acc_2_v1x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v1x0_typed, align 4
  %acc_2_v1x0_v2 = bitcast <2 x half> %acc_2_v1x0_load to <2 x half>
  %acc_2_v1x0_sram_e0 = extractelement <2 x half> %acc_2_v1x0_v2, i32 0
  %acc_2_v1x0_sram_e1 = extractelement <2 x half> %acc_2_v1x0_v2, i32 1
  %acc_2_v1x0_sram_v0 = insertelement <64 x half> undef, half %acc_2_v1x0_sram_e0, i32 0
  %acc_2_v1x0_sram = insertelement <64 x half> %acc_2_v1x0_sram_v0, half %acc_2_v1x0_sram_e1, i32 1
  %acc_2_c1d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_1, <64 x half> %acc_2_v1x0_sram, <64 x float> %acc_2_c0d0) #3
  %acc_2_v1x1_row = add i32 %morton_y, 8
  %acc_2_v1x1_col = add i32 %morton_x, 8
  %acc_2_v1x1_addr = mul i32 %acc_2_v1x1_row, 16
  %acc_2_v1x1_addr2 = add i32 %acc_2_v1x1_addr, %acc_2_v1x1_col
  %acc_2_v1x1_byte = mul i32 %acc_2_v1x1_addr2, 2
  %acc_2_v1x1_byte64 = zext i32 %acc_2_v1x1_byte to i64
  %acc_2_v1x1_byte64o = add i64 %acc_2_v1x1_byte64, 2048
  %acc_2_v1x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v1x1_byte64o
  %acc_2_v1x1_typed = bitcast i8 addrspace(3)* %acc_2_v1x1_ptr to <2 x half> addrspace(3)*
  %acc_2_v1x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v1x1_typed, align 4
  %acc_2_v1x1_v2 = bitcast <2 x half> %acc_2_v1x1_load to <2 x half>
  %acc_2_v1x1_sram_e0 = extractelement <2 x half> %acc_2_v1x1_v2, i32 0
  %acc_2_v1x1_sram_e1 = extractelement <2 x half> %acc_2_v1x1_v2, i32 1
  %acc_2_v1x1_sram_v0 = insertelement <64 x half> undef, half %acc_2_v1x1_sram_e0, i32 0
  %acc_2_v1x1_sram = insertelement <64 x half> %acc_2_v1x1_sram_v0, half %acc_2_v1x1_sram_e1, i32 1
  %acc_2_c1d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_1, <64 x half> %acc_2_v1x1_sram, <64 x float> %acc_2_c0d1) #3
  %acc_2_v2x0_row = add i32 %morton_y, 16
  %acc_2_v2x0_col = add i32 %morton_x, 0
  %acc_2_v2x0_addr = mul i32 %acc_2_v2x0_row, 16
  %acc_2_v2x0_addr2 = add i32 %acc_2_v2x0_addr, %acc_2_v2x0_col
  %acc_2_v2x0_byte = mul i32 %acc_2_v2x0_addr2, 2
  %acc_2_v2x0_byte64 = zext i32 %acc_2_v2x0_byte to i64
  %acc_2_v2x0_byte64o = add i64 %acc_2_v2x0_byte64, 2048
  %acc_2_v2x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v2x0_byte64o
  %acc_2_v2x0_typed = bitcast i8 addrspace(3)* %acc_2_v2x0_ptr to <2 x half> addrspace(3)*
  %acc_2_v2x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v2x0_typed, align 4
  %acc_2_v2x0_v2 = bitcast <2 x half> %acc_2_v2x0_load to <2 x half>
  %acc_2_v2x0_sram_e0 = extractelement <2 x half> %acc_2_v2x0_v2, i32 0
  %acc_2_v2x0_sram_e1 = extractelement <2 x half> %acc_2_v2x0_v2, i32 1
  %acc_2_v2x0_sram_v0 = insertelement <64 x half> undef, half %acc_2_v2x0_sram_e0, i32 0
  %acc_2_v2x0_sram = insertelement <64 x half> %acc_2_v2x0_sram_v0, half %acc_2_v2x0_sram_e1, i32 1
  %acc_2_c2d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_2, <64 x half> %acc_2_v2x0_sram, <64 x float> %acc_2_c1d0) #3
  %acc_2_v2x1_row = add i32 %morton_y, 16
  %acc_2_v2x1_col = add i32 %morton_x, 8
  %acc_2_v2x1_addr = mul i32 %acc_2_v2x1_row, 16
  %acc_2_v2x1_addr2 = add i32 %acc_2_v2x1_addr, %acc_2_v2x1_col
  %acc_2_v2x1_byte = mul i32 %acc_2_v2x1_addr2, 2
  %acc_2_v2x1_byte64 = zext i32 %acc_2_v2x1_byte to i64
  %acc_2_v2x1_byte64o = add i64 %acc_2_v2x1_byte64, 2048
  %acc_2_v2x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v2x1_byte64o
  %acc_2_v2x1_typed = bitcast i8 addrspace(3)* %acc_2_v2x1_ptr to <2 x half> addrspace(3)*
  %acc_2_v2x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v2x1_typed, align 4
  %acc_2_v2x1_v2 = bitcast <2 x half> %acc_2_v2x1_load to <2 x half>
  %acc_2_v2x1_sram_e0 = extractelement <2 x half> %acc_2_v2x1_v2, i32 0
  %acc_2_v2x1_sram_e1 = extractelement <2 x half> %acc_2_v2x1_v2, i32 1
  %acc_2_v2x1_sram_v0 = insertelement <64 x half> undef, half %acc_2_v2x1_sram_e0, i32 0
  %acc_2_v2x1_sram = insertelement <64 x half> %acc_2_v2x1_sram_v0, half %acc_2_v2x1_sram_e1, i32 1
  %acc_2_c2d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_2, <64 x half> %acc_2_v2x1_sram, <64 x float> %acc_2_c1d1) #3
  %acc_2_v3x0_row = add i32 %morton_y, 24
  %acc_2_v3x0_col = add i32 %morton_x, 0
  %acc_2_v3x0_addr = mul i32 %acc_2_v3x0_row, 16
  %acc_2_v3x0_addr2 = add i32 %acc_2_v3x0_addr, %acc_2_v3x0_col
  %acc_2_v3x0_byte = mul i32 %acc_2_v3x0_addr2, 2
  %acc_2_v3x0_byte64 = zext i32 %acc_2_v3x0_byte to i64
  %acc_2_v3x0_byte64o = add i64 %acc_2_v3x0_byte64, 2048
  %acc_2_v3x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v3x0_byte64o
  %acc_2_v3x0_typed = bitcast i8 addrspace(3)* %acc_2_v3x0_ptr to <2 x half> addrspace(3)*
  %acc_2_v3x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v3x0_typed, align 4
  %acc_2_v3x0_v2 = bitcast <2 x half> %acc_2_v3x0_load to <2 x half>
  %acc_2_v3x0_sram_e0 = extractelement <2 x half> %acc_2_v3x0_v2, i32 0
  %acc_2_v3x0_sram_e1 = extractelement <2 x half> %acc_2_v3x0_v2, i32 1
  %acc_2_v3x0_sram_v0 = insertelement <64 x half> undef, half %acc_2_v3x0_sram_e0, i32 0
  %acc_2_v3x0_sram = insertelement <64 x half> %acc_2_v3x0_sram_v0, half %acc_2_v3x0_sram_e1, i32 1
  %acc_2_c3d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_3, <64 x half> %acc_2_v3x0_sram, <64 x float> %acc_2_c2d0) #3
  %acc_2_v3x1_row = add i32 %morton_y, 24
  %acc_2_v3x1_col = add i32 %morton_x, 8
  %acc_2_v3x1_addr = mul i32 %acc_2_v3x1_row, 16
  %acc_2_v3x1_addr2 = add i32 %acc_2_v3x1_addr, %acc_2_v3x1_col
  %acc_2_v3x1_byte = mul i32 %acc_2_v3x1_addr2, 2
  %acc_2_v3x1_byte64 = zext i32 %acc_2_v3x1_byte to i64
  %acc_2_v3x1_byte64o = add i64 %acc_2_v3x1_byte64, 2048
  %acc_2_v3x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v3x1_byte64o
  %acc_2_v3x1_typed = bitcast i8 addrspace(3)* %acc_2_v3x1_ptr to <2 x half> addrspace(3)*
  %acc_2_v3x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v3x1_typed, align 4
  %acc_2_v3x1_v2 = bitcast <2 x half> %acc_2_v3x1_load to <2 x half>
  %acc_2_v3x1_sram_e0 = extractelement <2 x half> %acc_2_v3x1_v2, i32 0
  %acc_2_v3x1_sram_e1 = extractelement <2 x half> %acc_2_v3x1_v2, i32 1
  %acc_2_v3x1_sram_v0 = insertelement <64 x half> undef, half %acc_2_v3x1_sram_e0, i32 0
  %acc_2_v3x1_sram = insertelement <64 x half> %acc_2_v3x1_sram_v0, half %acc_2_v3x1_sram_e1, i32 1
  %acc_2_c3d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_3, <64 x half> %acc_2_v3x1_sram, <64 x float> %acc_2_c2d1) #3
  %acc_2_v4x0_row = add i32 %morton_y, 32
  %acc_2_v4x0_col = add i32 %morton_x, 0
  %acc_2_v4x0_addr = mul i32 %acc_2_v4x0_row, 16
  %acc_2_v4x0_addr2 = add i32 %acc_2_v4x0_addr, %acc_2_v4x0_col
  %acc_2_v4x0_byte = mul i32 %acc_2_v4x0_addr2, 2
  %acc_2_v4x0_byte64 = zext i32 %acc_2_v4x0_byte to i64
  %acc_2_v4x0_byte64o = add i64 %acc_2_v4x0_byte64, 2048
  %acc_2_v4x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v4x0_byte64o
  %acc_2_v4x0_typed = bitcast i8 addrspace(3)* %acc_2_v4x0_ptr to <2 x half> addrspace(3)*
  %acc_2_v4x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v4x0_typed, align 4
  %acc_2_v4x0_v2 = bitcast <2 x half> %acc_2_v4x0_load to <2 x half>
  %acc_2_v4x0_sram_e0 = extractelement <2 x half> %acc_2_v4x0_v2, i32 0
  %acc_2_v4x0_sram_e1 = extractelement <2 x half> %acc_2_v4x0_v2, i32 1
  %acc_2_v4x0_sram_v0 = insertelement <64 x half> undef, half %acc_2_v4x0_sram_e0, i32 0
  %acc_2_v4x0_sram = insertelement <64 x half> %acc_2_v4x0_sram_v0, half %acc_2_v4x0_sram_e1, i32 1
  %acc_2_c4d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_4, <64 x half> %acc_2_v4x0_sram, <64 x float> %acc_2_c3d0) #3
  %acc_2_v4x1_row = add i32 %morton_y, 32
  %acc_2_v4x1_col = add i32 %morton_x, 8
  %acc_2_v4x1_addr = mul i32 %acc_2_v4x1_row, 16
  %acc_2_v4x1_addr2 = add i32 %acc_2_v4x1_addr, %acc_2_v4x1_col
  %acc_2_v4x1_byte = mul i32 %acc_2_v4x1_addr2, 2
  %acc_2_v4x1_byte64 = zext i32 %acc_2_v4x1_byte to i64
  %acc_2_v4x1_byte64o = add i64 %acc_2_v4x1_byte64, 2048
  %acc_2_v4x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v4x1_byte64o
  %acc_2_v4x1_typed = bitcast i8 addrspace(3)* %acc_2_v4x1_ptr to <2 x half> addrspace(3)*
  %acc_2_v4x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v4x1_typed, align 4
  %acc_2_v4x1_v2 = bitcast <2 x half> %acc_2_v4x1_load to <2 x half>
  %acc_2_v4x1_sram_e0 = extractelement <2 x half> %acc_2_v4x1_v2, i32 0
  %acc_2_v4x1_sram_e1 = extractelement <2 x half> %acc_2_v4x1_v2, i32 1
  %acc_2_v4x1_sram_v0 = insertelement <64 x half> undef, half %acc_2_v4x1_sram_e0, i32 0
  %acc_2_v4x1_sram = insertelement <64 x half> %acc_2_v4x1_sram_v0, half %acc_2_v4x1_sram_e1, i32 1
  %acc_2_c4d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_4, <64 x half> %acc_2_v4x1_sram, <64 x float> %acc_2_c3d1) #3
  %acc_2_v5x0_row = add i32 %morton_y, 40
  %acc_2_v5x0_col = add i32 %morton_x, 0
  %acc_2_v5x0_addr = mul i32 %acc_2_v5x0_row, 16
  %acc_2_v5x0_addr2 = add i32 %acc_2_v5x0_addr, %acc_2_v5x0_col
  %acc_2_v5x0_byte = mul i32 %acc_2_v5x0_addr2, 2
  %acc_2_v5x0_byte64 = zext i32 %acc_2_v5x0_byte to i64
  %acc_2_v5x0_byte64o = add i64 %acc_2_v5x0_byte64, 2048
  %acc_2_v5x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v5x0_byte64o
  %acc_2_v5x0_typed = bitcast i8 addrspace(3)* %acc_2_v5x0_ptr to <2 x half> addrspace(3)*
  %acc_2_v5x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v5x0_typed, align 4
  %acc_2_v5x0_v2 = bitcast <2 x half> %acc_2_v5x0_load to <2 x half>
  %acc_2_v5x0_sram_e0 = extractelement <2 x half> %acc_2_v5x0_v2, i32 0
  %acc_2_v5x0_sram_e1 = extractelement <2 x half> %acc_2_v5x0_v2, i32 1
  %acc_2_v5x0_sram_v0 = insertelement <64 x half> undef, half %acc_2_v5x0_sram_e0, i32 0
  %acc_2_v5x0_sram = insertelement <64 x half> %acc_2_v5x0_sram_v0, half %acc_2_v5x0_sram_e1, i32 1
  %acc_2_c5d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_5, <64 x half> %acc_2_v5x0_sram, <64 x float> %acc_2_c4d0) #3
  %acc_2_v5x1_row = add i32 %morton_y, 40
  %acc_2_v5x1_col = add i32 %morton_x, 8
  %acc_2_v5x1_addr = mul i32 %acc_2_v5x1_row, 16
  %acc_2_v5x1_addr2 = add i32 %acc_2_v5x1_addr, %acc_2_v5x1_col
  %acc_2_v5x1_byte = mul i32 %acc_2_v5x1_addr2, 2
  %acc_2_v5x1_byte64 = zext i32 %acc_2_v5x1_byte to i64
  %acc_2_v5x1_byte64o = add i64 %acc_2_v5x1_byte64, 2048
  %acc_2_v5x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v5x1_byte64o
  %acc_2_v5x1_typed = bitcast i8 addrspace(3)* %acc_2_v5x1_ptr to <2 x half> addrspace(3)*
  %acc_2_v5x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v5x1_typed, align 4
  %acc_2_v5x1_v2 = bitcast <2 x half> %acc_2_v5x1_load to <2 x half>
  %acc_2_v5x1_sram_e0 = extractelement <2 x half> %acc_2_v5x1_v2, i32 0
  %acc_2_v5x1_sram_e1 = extractelement <2 x half> %acc_2_v5x1_v2, i32 1
  %acc_2_v5x1_sram_v0 = insertelement <64 x half> undef, half %acc_2_v5x1_sram_e0, i32 0
  %acc_2_v5x1_sram = insertelement <64 x half> %acc_2_v5x1_sram_v0, half %acc_2_v5x1_sram_e1, i32 1
  %acc_2_c5d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_5, <64 x half> %acc_2_v5x1_sram, <64 x float> %acc_2_c4d1) #3
  %acc_2_v6x0_row = add i32 %morton_y, 48
  %acc_2_v6x0_col = add i32 %morton_x, 0
  %acc_2_v6x0_addr = mul i32 %acc_2_v6x0_row, 16
  %acc_2_v6x0_addr2 = add i32 %acc_2_v6x0_addr, %acc_2_v6x0_col
  %acc_2_v6x0_byte = mul i32 %acc_2_v6x0_addr2, 2
  %acc_2_v6x0_byte64 = zext i32 %acc_2_v6x0_byte to i64
  %acc_2_v6x0_byte64o = add i64 %acc_2_v6x0_byte64, 2048
  %acc_2_v6x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v6x0_byte64o
  %acc_2_v6x0_typed = bitcast i8 addrspace(3)* %acc_2_v6x0_ptr to <2 x half> addrspace(3)*
  %acc_2_v6x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v6x0_typed, align 4
  %acc_2_v6x0_v2 = bitcast <2 x half> %acc_2_v6x0_load to <2 x half>
  %acc_2_v6x0_sram_e0 = extractelement <2 x half> %acc_2_v6x0_v2, i32 0
  %acc_2_v6x0_sram_e1 = extractelement <2 x half> %acc_2_v6x0_v2, i32 1
  %acc_2_v6x0_sram_v0 = insertelement <64 x half> undef, half %acc_2_v6x0_sram_e0, i32 0
  %acc_2_v6x0_sram = insertelement <64 x half> %acc_2_v6x0_sram_v0, half %acc_2_v6x0_sram_e1, i32 1
  %acc_2_c6d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_6, <64 x half> %acc_2_v6x0_sram, <64 x float> %acc_2_c5d0) #3
  %acc_2_v6x1_row = add i32 %morton_y, 48
  %acc_2_v6x1_col = add i32 %morton_x, 8
  %acc_2_v6x1_addr = mul i32 %acc_2_v6x1_row, 16
  %acc_2_v6x1_addr2 = add i32 %acc_2_v6x1_addr, %acc_2_v6x1_col
  %acc_2_v6x1_byte = mul i32 %acc_2_v6x1_addr2, 2
  %acc_2_v6x1_byte64 = zext i32 %acc_2_v6x1_byte to i64
  %acc_2_v6x1_byte64o = add i64 %acc_2_v6x1_byte64, 2048
  %acc_2_v6x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v6x1_byte64o
  %acc_2_v6x1_typed = bitcast i8 addrspace(3)* %acc_2_v6x1_ptr to <2 x half> addrspace(3)*
  %acc_2_v6x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v6x1_typed, align 4
  %acc_2_v6x1_v2 = bitcast <2 x half> %acc_2_v6x1_load to <2 x half>
  %acc_2_v6x1_sram_e0 = extractelement <2 x half> %acc_2_v6x1_v2, i32 0
  %acc_2_v6x1_sram_e1 = extractelement <2 x half> %acc_2_v6x1_v2, i32 1
  %acc_2_v6x1_sram_v0 = insertelement <64 x half> undef, half %acc_2_v6x1_sram_e0, i32 0
  %acc_2_v6x1_sram = insertelement <64 x half> %acc_2_v6x1_sram_v0, half %acc_2_v6x1_sram_e1, i32 1
  %acc_2_c6d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_6, <64 x half> %acc_2_v6x1_sram, <64 x float> %acc_2_c5d1) #3
  %acc_2_v7x0_row = add i32 %morton_y, 56
  %acc_2_v7x0_col = add i32 %morton_x, 0
  %acc_2_v7x0_addr = mul i32 %acc_2_v7x0_row, 16
  %acc_2_v7x0_addr2 = add i32 %acc_2_v7x0_addr, %acc_2_v7x0_col
  %acc_2_v7x0_byte = mul i32 %acc_2_v7x0_addr2, 2
  %acc_2_v7x0_byte64 = zext i32 %acc_2_v7x0_byte to i64
  %acc_2_v7x0_byte64o = add i64 %acc_2_v7x0_byte64, 2048
  %acc_2_v7x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v7x0_byte64o
  %acc_2_v7x0_typed = bitcast i8 addrspace(3)* %acc_2_v7x0_ptr to <2 x half> addrspace(3)*
  %acc_2_v7x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v7x0_typed, align 4
  %acc_2_v7x0_v2 = bitcast <2 x half> %acc_2_v7x0_load to <2 x half>
  %acc_2_v7x0_sram_e0 = extractelement <2 x half> %acc_2_v7x0_v2, i32 0
  %acc_2_v7x0_sram_e1 = extractelement <2 x half> %acc_2_v7x0_v2, i32 1
  %acc_2_v7x0_sram_v0 = insertelement <64 x half> undef, half %acc_2_v7x0_sram_e0, i32 0
  %acc_2_v7x0_sram = insertelement <64 x half> %acc_2_v7x0_sram_v0, half %acc_2_v7x0_sram_e1, i32 1
  %acc_2_c7d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_7, <64 x half> %acc_2_v7x0_sram, <64 x float> %acc_2_c6d0) #3
  %acc_2_v7x1_row = add i32 %morton_y, 56
  %acc_2_v7x1_col = add i32 %morton_x, 8
  %acc_2_v7x1_addr = mul i32 %acc_2_v7x1_row, 16
  %acc_2_v7x1_addr2 = add i32 %acc_2_v7x1_addr, %acc_2_v7x1_col
  %acc_2_v7x1_byte = mul i32 %acc_2_v7x1_addr2, 2
  %acc_2_v7x1_byte64 = zext i32 %acc_2_v7x1_byte to i64
  %acc_2_v7x1_byte64o = add i64 %acc_2_v7x1_byte64, 2048
  %acc_2_v7x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_2_v7x1_byte64o
  %acc_2_v7x1_typed = bitcast i8 addrspace(3)* %acc_2_v7x1_ptr to <2 x half> addrspace(3)*
  %acc_2_v7x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_2_v7x1_typed, align 4
  %acc_2_v7x1_v2 = bitcast <2 x half> %acc_2_v7x1_load to <2 x half>
  %acc_2_v7x1_sram_e0 = extractelement <2 x half> %acc_2_v7x1_v2, i32 0
  %acc_2_v7x1_sram_e1 = extractelement <2 x half> %acc_2_v7x1_v2, i32 1
  %acc_2_v7x1_sram_v0 = insertelement <64 x half> undef, half %acc_2_v7x1_sram_e0, i32 0
  %acc_2_v7x1_sram = insertelement <64 x half> %acc_2_v7x1_sram_v0, half %acc_2_v7x1_sram_e1, i32 1
  %acc_2_c7d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_7, <64 x half> %acc_2_v7x1_sram, <64 x float> %acc_2_c6d1) #3
  call void @air.wg.barrier(i32 2, i32 1)

  ; === Sync copy (acc_3_v) — all threads cooperative ===
  %acc_3_vsc_t0 = mul i32 %sidx, 32
  %acc_3_vsc_tid = add i32 %acc_3_vsc_t0, %lane_id
  %acc_3_vsc_drem = sub i32 64, 48
  %acc_3_vsc_dcmp = icmp ult i32 %acc_3_vsc_drem, 16
  %acc_3_vsc_dsrc = select i1 %acc_3_vsc_dcmp, i32 %acc_3_vsc_drem, i32 16
  %acc_3_vsc_srem = sub i32 64, %c
  %acc_3_vsc_scmp = icmp ult i32 %acc_3_vsc_srem, 64
  %acc_3_vsc_ssrc = select i1 %acc_3_vsc_scmp, i32 %acc_3_vsc_srem, i32 64
  br label %acc_3_vsc_pre

acc_3_vsc_pre:
  br label %acc_3_vsc_hdr

acc_3_vsc_hdr:
  %acc_3_vsc_i = phi i32 [%acc_3_vsc_tid, %acc_3_vsc_pre], [%acc_3_vsc_inx, %acc_3_vsc_st]
  %acc_3_vsc_done = icmp uge i32 %acc_3_vsc_i, 1024
  br i1 %acc_3_vsc_done, label %acc_3_vsc_end, label %acc_3_vsc_body

acc_3_vsc_body:
  %acc_3_vsc_row = lshr i32 %acc_3_vsc_i, 4
  %acc_3_vsc_col = and i32 %acc_3_vsc_i, 15
  %acc_3_vsc_rok = icmp ult i32 %acc_3_vsc_row, %acc_3_vsc_ssrc
  %acc_3_vsc_cok = icmp ult i32 %acc_3_vsc_col, %acc_3_vsc_dsrc
  %acc_3_vsc_ib = and i1 %acc_3_vsc_rok, %acc_3_vsc_cok
  br i1 %acc_3_vsc_ib, label %acc_3_vsc_ld, label %acc_3_vsc_zr

acc_3_vsc_ld:
  %acc_3_vsc_sr = add i32 %c, %acc_3_vsc_row
  %acc_3_vsc_sa = mul i32 %acc_3_vsc_sr, 64
  %acc_3_vsc_sc = add i32 48, %acc_3_vsc_col
  %acc_3_vsc_sad = add i32 %acc_3_vsc_sa, %acc_3_vsc_sc
  %acc_3_vsc_soff = zext i32 %acc_3_vsc_sad to i64
  %acc_3_vsc_sbyt = mul i64 %acc_3_vsc_soff, 2
  %acc_3_vsc_sp = getelementptr i8, i8 addrspace(1)* %V, i64 %acc_3_vsc_sbyt
  %acc_3_vsc_spt = bitcast i8 addrspace(1)* %acc_3_vsc_sp to i16 addrspace(1)*
  %acc_3_vsc_lv = load i16, i16 addrspace(1)* %acc_3_vsc_spt
  br label %acc_3_vsc_st

acc_3_vsc_zr:
  br label %acc_3_vsc_st

acc_3_vsc_st:
  %acc_3_vsc_val = phi i16 [%acc_3_vsc_lv, %acc_3_vsc_ld], [0, %acc_3_vsc_zr]
  %acc_3_vsc_tr = mul i32 %acc_3_vsc_row, 16
  %acc_3_vsc_ta = add i32 %acc_3_vsc_tr, %acc_3_vsc_col
  %acc_3_vsc_tb = mul i32 %acc_3_vsc_ta, 2
  %acc_3_vsc_tb64 = zext i32 %acc_3_vsc_tb to i64
  %acc_3_vsc_tb64o = add i64 %acc_3_vsc_tb64, 2048
  %acc_3_vsc_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_vsc_tb64o
  %acc_3_vsc_tpt = bitcast i8 addrspace(3)* %acc_3_vsc_tp to i16 addrspace(3)*
  store i16 %acc_3_vsc_val, i16 addrspace(3)* %acc_3_vsc_tpt
  %acc_3_vsc_inx = add i32 %acc_3_vsc_i, 64
  br label %acc_3_vsc_hdr

acc_3_vsc_end:
  call void @air.wg.barrier(i32 2, i32 1)

  %acc_3_v0x0_row = add i32 %morton_y, 0
  %acc_3_v0x0_col = add i32 %morton_x, 0
  %acc_3_v0x0_addr = mul i32 %acc_3_v0x0_row, 16
  %acc_3_v0x0_addr2 = add i32 %acc_3_v0x0_addr, %acc_3_v0x0_col
  %acc_3_v0x0_byte = mul i32 %acc_3_v0x0_addr2, 2
  %acc_3_v0x0_byte64 = zext i32 %acc_3_v0x0_byte to i64
  %acc_3_v0x0_byte64o = add i64 %acc_3_v0x0_byte64, 2048
  %acc_3_v0x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v0x0_byte64o
  %acc_3_v0x0_typed = bitcast i8 addrspace(3)* %acc_3_v0x0_ptr to <2 x half> addrspace(3)*
  %acc_3_v0x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v0x0_typed, align 4
  %acc_3_v0x0_v2 = bitcast <2 x half> %acc_3_v0x0_load to <2 x half>
  %acc_3_v0x0_sram_e0 = extractelement <2 x half> %acc_3_v0x0_v2, i32 0
  %acc_3_v0x0_sram_e1 = extractelement <2 x half> %acc_3_v0x0_v2, i32 1
  %acc_3_v0x0_sram_v0 = insertelement <64 x half> undef, half %acc_3_v0x0_sram_e0, i32 0
  %acc_3_v0x0_sram = insertelement <64 x half> %acc_3_v0x0_sram_v0, half %acc_3_v0x0_sram_e1, i32 1
  %acc_3_c0d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_0, <64 x half> %acc_3_v0x0_sram, <64 x float> %acc_corrected_6) #3
  %acc_3_v0x1_row = add i32 %morton_y, 0
  %acc_3_v0x1_col = add i32 %morton_x, 8
  %acc_3_v0x1_addr = mul i32 %acc_3_v0x1_row, 16
  %acc_3_v0x1_addr2 = add i32 %acc_3_v0x1_addr, %acc_3_v0x1_col
  %acc_3_v0x1_byte = mul i32 %acc_3_v0x1_addr2, 2
  %acc_3_v0x1_byte64 = zext i32 %acc_3_v0x1_byte to i64
  %acc_3_v0x1_byte64o = add i64 %acc_3_v0x1_byte64, 2048
  %acc_3_v0x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v0x1_byte64o
  %acc_3_v0x1_typed = bitcast i8 addrspace(3)* %acc_3_v0x1_ptr to <2 x half> addrspace(3)*
  %acc_3_v0x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v0x1_typed, align 4
  %acc_3_v0x1_v2 = bitcast <2 x half> %acc_3_v0x1_load to <2 x half>
  %acc_3_v0x1_sram_e0 = extractelement <2 x half> %acc_3_v0x1_v2, i32 0
  %acc_3_v0x1_sram_e1 = extractelement <2 x half> %acc_3_v0x1_v2, i32 1
  %acc_3_v0x1_sram_v0 = insertelement <64 x half> undef, half %acc_3_v0x1_sram_e0, i32 0
  %acc_3_v0x1_sram = insertelement <64 x half> %acc_3_v0x1_sram_v0, half %acc_3_v0x1_sram_e1, i32 1
  %acc_3_c0d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_0, <64 x half> %acc_3_v0x1_sram, <64 x float> %acc_corrected_7) #3
  %acc_3_v1x0_row = add i32 %morton_y, 8
  %acc_3_v1x0_col = add i32 %morton_x, 0
  %acc_3_v1x0_addr = mul i32 %acc_3_v1x0_row, 16
  %acc_3_v1x0_addr2 = add i32 %acc_3_v1x0_addr, %acc_3_v1x0_col
  %acc_3_v1x0_byte = mul i32 %acc_3_v1x0_addr2, 2
  %acc_3_v1x0_byte64 = zext i32 %acc_3_v1x0_byte to i64
  %acc_3_v1x0_byte64o = add i64 %acc_3_v1x0_byte64, 2048
  %acc_3_v1x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v1x0_byte64o
  %acc_3_v1x0_typed = bitcast i8 addrspace(3)* %acc_3_v1x0_ptr to <2 x half> addrspace(3)*
  %acc_3_v1x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v1x0_typed, align 4
  %acc_3_v1x0_v2 = bitcast <2 x half> %acc_3_v1x0_load to <2 x half>
  %acc_3_v1x0_sram_e0 = extractelement <2 x half> %acc_3_v1x0_v2, i32 0
  %acc_3_v1x0_sram_e1 = extractelement <2 x half> %acc_3_v1x0_v2, i32 1
  %acc_3_v1x0_sram_v0 = insertelement <64 x half> undef, half %acc_3_v1x0_sram_e0, i32 0
  %acc_3_v1x0_sram = insertelement <64 x half> %acc_3_v1x0_sram_v0, half %acc_3_v1x0_sram_e1, i32 1
  %acc_3_c1d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_1, <64 x half> %acc_3_v1x0_sram, <64 x float> %acc_3_c0d0) #3
  %acc_3_v1x1_row = add i32 %morton_y, 8
  %acc_3_v1x1_col = add i32 %morton_x, 8
  %acc_3_v1x1_addr = mul i32 %acc_3_v1x1_row, 16
  %acc_3_v1x1_addr2 = add i32 %acc_3_v1x1_addr, %acc_3_v1x1_col
  %acc_3_v1x1_byte = mul i32 %acc_3_v1x1_addr2, 2
  %acc_3_v1x1_byte64 = zext i32 %acc_3_v1x1_byte to i64
  %acc_3_v1x1_byte64o = add i64 %acc_3_v1x1_byte64, 2048
  %acc_3_v1x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v1x1_byte64o
  %acc_3_v1x1_typed = bitcast i8 addrspace(3)* %acc_3_v1x1_ptr to <2 x half> addrspace(3)*
  %acc_3_v1x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v1x1_typed, align 4
  %acc_3_v1x1_v2 = bitcast <2 x half> %acc_3_v1x1_load to <2 x half>
  %acc_3_v1x1_sram_e0 = extractelement <2 x half> %acc_3_v1x1_v2, i32 0
  %acc_3_v1x1_sram_e1 = extractelement <2 x half> %acc_3_v1x1_v2, i32 1
  %acc_3_v1x1_sram_v0 = insertelement <64 x half> undef, half %acc_3_v1x1_sram_e0, i32 0
  %acc_3_v1x1_sram = insertelement <64 x half> %acc_3_v1x1_sram_v0, half %acc_3_v1x1_sram_e1, i32 1
  %acc_3_c1d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_1, <64 x half> %acc_3_v1x1_sram, <64 x float> %acc_3_c0d1) #3
  %acc_3_v2x0_row = add i32 %morton_y, 16
  %acc_3_v2x0_col = add i32 %morton_x, 0
  %acc_3_v2x0_addr = mul i32 %acc_3_v2x0_row, 16
  %acc_3_v2x0_addr2 = add i32 %acc_3_v2x0_addr, %acc_3_v2x0_col
  %acc_3_v2x0_byte = mul i32 %acc_3_v2x0_addr2, 2
  %acc_3_v2x0_byte64 = zext i32 %acc_3_v2x0_byte to i64
  %acc_3_v2x0_byte64o = add i64 %acc_3_v2x0_byte64, 2048
  %acc_3_v2x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v2x0_byte64o
  %acc_3_v2x0_typed = bitcast i8 addrspace(3)* %acc_3_v2x0_ptr to <2 x half> addrspace(3)*
  %acc_3_v2x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v2x0_typed, align 4
  %acc_3_v2x0_v2 = bitcast <2 x half> %acc_3_v2x0_load to <2 x half>
  %acc_3_v2x0_sram_e0 = extractelement <2 x half> %acc_3_v2x0_v2, i32 0
  %acc_3_v2x0_sram_e1 = extractelement <2 x half> %acc_3_v2x0_v2, i32 1
  %acc_3_v2x0_sram_v0 = insertelement <64 x half> undef, half %acc_3_v2x0_sram_e0, i32 0
  %acc_3_v2x0_sram = insertelement <64 x half> %acc_3_v2x0_sram_v0, half %acc_3_v2x0_sram_e1, i32 1
  %acc_3_c2d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_2, <64 x half> %acc_3_v2x0_sram, <64 x float> %acc_3_c1d0) #3
  %acc_3_v2x1_row = add i32 %morton_y, 16
  %acc_3_v2x1_col = add i32 %morton_x, 8
  %acc_3_v2x1_addr = mul i32 %acc_3_v2x1_row, 16
  %acc_3_v2x1_addr2 = add i32 %acc_3_v2x1_addr, %acc_3_v2x1_col
  %acc_3_v2x1_byte = mul i32 %acc_3_v2x1_addr2, 2
  %acc_3_v2x1_byte64 = zext i32 %acc_3_v2x1_byte to i64
  %acc_3_v2x1_byte64o = add i64 %acc_3_v2x1_byte64, 2048
  %acc_3_v2x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v2x1_byte64o
  %acc_3_v2x1_typed = bitcast i8 addrspace(3)* %acc_3_v2x1_ptr to <2 x half> addrspace(3)*
  %acc_3_v2x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v2x1_typed, align 4
  %acc_3_v2x1_v2 = bitcast <2 x half> %acc_3_v2x1_load to <2 x half>
  %acc_3_v2x1_sram_e0 = extractelement <2 x half> %acc_3_v2x1_v2, i32 0
  %acc_3_v2x1_sram_e1 = extractelement <2 x half> %acc_3_v2x1_v2, i32 1
  %acc_3_v2x1_sram_v0 = insertelement <64 x half> undef, half %acc_3_v2x1_sram_e0, i32 0
  %acc_3_v2x1_sram = insertelement <64 x half> %acc_3_v2x1_sram_v0, half %acc_3_v2x1_sram_e1, i32 1
  %acc_3_c2d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_2, <64 x half> %acc_3_v2x1_sram, <64 x float> %acc_3_c1d1) #3
  %acc_3_v3x0_row = add i32 %morton_y, 24
  %acc_3_v3x0_col = add i32 %morton_x, 0
  %acc_3_v3x0_addr = mul i32 %acc_3_v3x0_row, 16
  %acc_3_v3x0_addr2 = add i32 %acc_3_v3x0_addr, %acc_3_v3x0_col
  %acc_3_v3x0_byte = mul i32 %acc_3_v3x0_addr2, 2
  %acc_3_v3x0_byte64 = zext i32 %acc_3_v3x0_byte to i64
  %acc_3_v3x0_byte64o = add i64 %acc_3_v3x0_byte64, 2048
  %acc_3_v3x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v3x0_byte64o
  %acc_3_v3x0_typed = bitcast i8 addrspace(3)* %acc_3_v3x0_ptr to <2 x half> addrspace(3)*
  %acc_3_v3x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v3x0_typed, align 4
  %acc_3_v3x0_v2 = bitcast <2 x half> %acc_3_v3x0_load to <2 x half>
  %acc_3_v3x0_sram_e0 = extractelement <2 x half> %acc_3_v3x0_v2, i32 0
  %acc_3_v3x0_sram_e1 = extractelement <2 x half> %acc_3_v3x0_v2, i32 1
  %acc_3_v3x0_sram_v0 = insertelement <64 x half> undef, half %acc_3_v3x0_sram_e0, i32 0
  %acc_3_v3x0_sram = insertelement <64 x half> %acc_3_v3x0_sram_v0, half %acc_3_v3x0_sram_e1, i32 1
  %acc_3_c3d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_3, <64 x half> %acc_3_v3x0_sram, <64 x float> %acc_3_c2d0) #3
  %acc_3_v3x1_row = add i32 %morton_y, 24
  %acc_3_v3x1_col = add i32 %morton_x, 8
  %acc_3_v3x1_addr = mul i32 %acc_3_v3x1_row, 16
  %acc_3_v3x1_addr2 = add i32 %acc_3_v3x1_addr, %acc_3_v3x1_col
  %acc_3_v3x1_byte = mul i32 %acc_3_v3x1_addr2, 2
  %acc_3_v3x1_byte64 = zext i32 %acc_3_v3x1_byte to i64
  %acc_3_v3x1_byte64o = add i64 %acc_3_v3x1_byte64, 2048
  %acc_3_v3x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v3x1_byte64o
  %acc_3_v3x1_typed = bitcast i8 addrspace(3)* %acc_3_v3x1_ptr to <2 x half> addrspace(3)*
  %acc_3_v3x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v3x1_typed, align 4
  %acc_3_v3x1_v2 = bitcast <2 x half> %acc_3_v3x1_load to <2 x half>
  %acc_3_v3x1_sram_e0 = extractelement <2 x half> %acc_3_v3x1_v2, i32 0
  %acc_3_v3x1_sram_e1 = extractelement <2 x half> %acc_3_v3x1_v2, i32 1
  %acc_3_v3x1_sram_v0 = insertelement <64 x half> undef, half %acc_3_v3x1_sram_e0, i32 0
  %acc_3_v3x1_sram = insertelement <64 x half> %acc_3_v3x1_sram_v0, half %acc_3_v3x1_sram_e1, i32 1
  %acc_3_c3d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_3, <64 x half> %acc_3_v3x1_sram, <64 x float> %acc_3_c2d1) #3
  %acc_3_v4x0_row = add i32 %morton_y, 32
  %acc_3_v4x0_col = add i32 %morton_x, 0
  %acc_3_v4x0_addr = mul i32 %acc_3_v4x0_row, 16
  %acc_3_v4x0_addr2 = add i32 %acc_3_v4x0_addr, %acc_3_v4x0_col
  %acc_3_v4x0_byte = mul i32 %acc_3_v4x0_addr2, 2
  %acc_3_v4x0_byte64 = zext i32 %acc_3_v4x0_byte to i64
  %acc_3_v4x0_byte64o = add i64 %acc_3_v4x0_byte64, 2048
  %acc_3_v4x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v4x0_byte64o
  %acc_3_v4x0_typed = bitcast i8 addrspace(3)* %acc_3_v4x0_ptr to <2 x half> addrspace(3)*
  %acc_3_v4x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v4x0_typed, align 4
  %acc_3_v4x0_v2 = bitcast <2 x half> %acc_3_v4x0_load to <2 x half>
  %acc_3_v4x0_sram_e0 = extractelement <2 x half> %acc_3_v4x0_v2, i32 0
  %acc_3_v4x0_sram_e1 = extractelement <2 x half> %acc_3_v4x0_v2, i32 1
  %acc_3_v4x0_sram_v0 = insertelement <64 x half> undef, half %acc_3_v4x0_sram_e0, i32 0
  %acc_3_v4x0_sram = insertelement <64 x half> %acc_3_v4x0_sram_v0, half %acc_3_v4x0_sram_e1, i32 1
  %acc_3_c4d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_4, <64 x half> %acc_3_v4x0_sram, <64 x float> %acc_3_c3d0) #3
  %acc_3_v4x1_row = add i32 %morton_y, 32
  %acc_3_v4x1_col = add i32 %morton_x, 8
  %acc_3_v4x1_addr = mul i32 %acc_3_v4x1_row, 16
  %acc_3_v4x1_addr2 = add i32 %acc_3_v4x1_addr, %acc_3_v4x1_col
  %acc_3_v4x1_byte = mul i32 %acc_3_v4x1_addr2, 2
  %acc_3_v4x1_byte64 = zext i32 %acc_3_v4x1_byte to i64
  %acc_3_v4x1_byte64o = add i64 %acc_3_v4x1_byte64, 2048
  %acc_3_v4x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v4x1_byte64o
  %acc_3_v4x1_typed = bitcast i8 addrspace(3)* %acc_3_v4x1_ptr to <2 x half> addrspace(3)*
  %acc_3_v4x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v4x1_typed, align 4
  %acc_3_v4x1_v2 = bitcast <2 x half> %acc_3_v4x1_load to <2 x half>
  %acc_3_v4x1_sram_e0 = extractelement <2 x half> %acc_3_v4x1_v2, i32 0
  %acc_3_v4x1_sram_e1 = extractelement <2 x half> %acc_3_v4x1_v2, i32 1
  %acc_3_v4x1_sram_v0 = insertelement <64 x half> undef, half %acc_3_v4x1_sram_e0, i32 0
  %acc_3_v4x1_sram = insertelement <64 x half> %acc_3_v4x1_sram_v0, half %acc_3_v4x1_sram_e1, i32 1
  %acc_3_c4d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_4, <64 x half> %acc_3_v4x1_sram, <64 x float> %acc_3_c3d1) #3
  %acc_3_v5x0_row = add i32 %morton_y, 40
  %acc_3_v5x0_col = add i32 %morton_x, 0
  %acc_3_v5x0_addr = mul i32 %acc_3_v5x0_row, 16
  %acc_3_v5x0_addr2 = add i32 %acc_3_v5x0_addr, %acc_3_v5x0_col
  %acc_3_v5x0_byte = mul i32 %acc_3_v5x0_addr2, 2
  %acc_3_v5x0_byte64 = zext i32 %acc_3_v5x0_byte to i64
  %acc_3_v5x0_byte64o = add i64 %acc_3_v5x0_byte64, 2048
  %acc_3_v5x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v5x0_byte64o
  %acc_3_v5x0_typed = bitcast i8 addrspace(3)* %acc_3_v5x0_ptr to <2 x half> addrspace(3)*
  %acc_3_v5x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v5x0_typed, align 4
  %acc_3_v5x0_v2 = bitcast <2 x half> %acc_3_v5x0_load to <2 x half>
  %acc_3_v5x0_sram_e0 = extractelement <2 x half> %acc_3_v5x0_v2, i32 0
  %acc_3_v5x0_sram_e1 = extractelement <2 x half> %acc_3_v5x0_v2, i32 1
  %acc_3_v5x0_sram_v0 = insertelement <64 x half> undef, half %acc_3_v5x0_sram_e0, i32 0
  %acc_3_v5x0_sram = insertelement <64 x half> %acc_3_v5x0_sram_v0, half %acc_3_v5x0_sram_e1, i32 1
  %acc_3_c5d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_5, <64 x half> %acc_3_v5x0_sram, <64 x float> %acc_3_c4d0) #3
  %acc_3_v5x1_row = add i32 %morton_y, 40
  %acc_3_v5x1_col = add i32 %morton_x, 8
  %acc_3_v5x1_addr = mul i32 %acc_3_v5x1_row, 16
  %acc_3_v5x1_addr2 = add i32 %acc_3_v5x1_addr, %acc_3_v5x1_col
  %acc_3_v5x1_byte = mul i32 %acc_3_v5x1_addr2, 2
  %acc_3_v5x1_byte64 = zext i32 %acc_3_v5x1_byte to i64
  %acc_3_v5x1_byte64o = add i64 %acc_3_v5x1_byte64, 2048
  %acc_3_v5x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v5x1_byte64o
  %acc_3_v5x1_typed = bitcast i8 addrspace(3)* %acc_3_v5x1_ptr to <2 x half> addrspace(3)*
  %acc_3_v5x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v5x1_typed, align 4
  %acc_3_v5x1_v2 = bitcast <2 x half> %acc_3_v5x1_load to <2 x half>
  %acc_3_v5x1_sram_e0 = extractelement <2 x half> %acc_3_v5x1_v2, i32 0
  %acc_3_v5x1_sram_e1 = extractelement <2 x half> %acc_3_v5x1_v2, i32 1
  %acc_3_v5x1_sram_v0 = insertelement <64 x half> undef, half %acc_3_v5x1_sram_e0, i32 0
  %acc_3_v5x1_sram = insertelement <64 x half> %acc_3_v5x1_sram_v0, half %acc_3_v5x1_sram_e1, i32 1
  %acc_3_c5d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_5, <64 x half> %acc_3_v5x1_sram, <64 x float> %acc_3_c4d1) #3
  %acc_3_v6x0_row = add i32 %morton_y, 48
  %acc_3_v6x0_col = add i32 %morton_x, 0
  %acc_3_v6x0_addr = mul i32 %acc_3_v6x0_row, 16
  %acc_3_v6x0_addr2 = add i32 %acc_3_v6x0_addr, %acc_3_v6x0_col
  %acc_3_v6x0_byte = mul i32 %acc_3_v6x0_addr2, 2
  %acc_3_v6x0_byte64 = zext i32 %acc_3_v6x0_byte to i64
  %acc_3_v6x0_byte64o = add i64 %acc_3_v6x0_byte64, 2048
  %acc_3_v6x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v6x0_byte64o
  %acc_3_v6x0_typed = bitcast i8 addrspace(3)* %acc_3_v6x0_ptr to <2 x half> addrspace(3)*
  %acc_3_v6x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v6x0_typed, align 4
  %acc_3_v6x0_v2 = bitcast <2 x half> %acc_3_v6x0_load to <2 x half>
  %acc_3_v6x0_sram_e0 = extractelement <2 x half> %acc_3_v6x0_v2, i32 0
  %acc_3_v6x0_sram_e1 = extractelement <2 x half> %acc_3_v6x0_v2, i32 1
  %acc_3_v6x0_sram_v0 = insertelement <64 x half> undef, half %acc_3_v6x0_sram_e0, i32 0
  %acc_3_v6x0_sram = insertelement <64 x half> %acc_3_v6x0_sram_v0, half %acc_3_v6x0_sram_e1, i32 1
  %acc_3_c6d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_6, <64 x half> %acc_3_v6x0_sram, <64 x float> %acc_3_c5d0) #3
  %acc_3_v6x1_row = add i32 %morton_y, 48
  %acc_3_v6x1_col = add i32 %morton_x, 8
  %acc_3_v6x1_addr = mul i32 %acc_3_v6x1_row, 16
  %acc_3_v6x1_addr2 = add i32 %acc_3_v6x1_addr, %acc_3_v6x1_col
  %acc_3_v6x1_byte = mul i32 %acc_3_v6x1_addr2, 2
  %acc_3_v6x1_byte64 = zext i32 %acc_3_v6x1_byte to i64
  %acc_3_v6x1_byte64o = add i64 %acc_3_v6x1_byte64, 2048
  %acc_3_v6x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v6x1_byte64o
  %acc_3_v6x1_typed = bitcast i8 addrspace(3)* %acc_3_v6x1_ptr to <2 x half> addrspace(3)*
  %acc_3_v6x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v6x1_typed, align 4
  %acc_3_v6x1_v2 = bitcast <2 x half> %acc_3_v6x1_load to <2 x half>
  %acc_3_v6x1_sram_e0 = extractelement <2 x half> %acc_3_v6x1_v2, i32 0
  %acc_3_v6x1_sram_e1 = extractelement <2 x half> %acc_3_v6x1_v2, i32 1
  %acc_3_v6x1_sram_v0 = insertelement <64 x half> undef, half %acc_3_v6x1_sram_e0, i32 0
  %acc_3_v6x1_sram = insertelement <64 x half> %acc_3_v6x1_sram_v0, half %acc_3_v6x1_sram_e1, i32 1
  %acc_3_c6d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_6, <64 x half> %acc_3_v6x1_sram, <64 x float> %acc_3_c5d1) #3
  %acc_3_v7x0_row = add i32 %morton_y, 56
  %acc_3_v7x0_col = add i32 %morton_x, 0
  %acc_3_v7x0_addr = mul i32 %acc_3_v7x0_row, 16
  %acc_3_v7x0_addr2 = add i32 %acc_3_v7x0_addr, %acc_3_v7x0_col
  %acc_3_v7x0_byte = mul i32 %acc_3_v7x0_addr2, 2
  %acc_3_v7x0_byte64 = zext i32 %acc_3_v7x0_byte to i64
  %acc_3_v7x0_byte64o = add i64 %acc_3_v7x0_byte64, 2048
  %acc_3_v7x0_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v7x0_byte64o
  %acc_3_v7x0_typed = bitcast i8 addrspace(3)* %acc_3_v7x0_ptr to <2 x half> addrspace(3)*
  %acc_3_v7x0_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v7x0_typed, align 4
  %acc_3_v7x0_v2 = bitcast <2 x half> %acc_3_v7x0_load to <2 x half>
  %acc_3_v7x0_sram_e0 = extractelement <2 x half> %acc_3_v7x0_v2, i32 0
  %acc_3_v7x0_sram_e1 = extractelement <2 x half> %acc_3_v7x0_v2, i32 1
  %acc_3_v7x0_sram_v0 = insertelement <64 x half> undef, half %acc_3_v7x0_sram_e0, i32 0
  %acc_3_v7x0_sram = insertelement <64 x half> %acc_3_v7x0_sram_v0, half %acc_3_v7x0_sram_e1, i32 1
  %acc_3_c7d0 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_7, <64 x half> %acc_3_v7x0_sram, <64 x float> %acc_3_c6d0) #3
  %acc_3_v7x1_row = add i32 %morton_y, 56
  %acc_3_v7x1_col = add i32 %morton_x, 8
  %acc_3_v7x1_addr = mul i32 %acc_3_v7x1_row, 16
  %acc_3_v7x1_addr2 = add i32 %acc_3_v7x1_addr, %acc_3_v7x1_col
  %acc_3_v7x1_byte = mul i32 %acc_3_v7x1_addr2, 2
  %acc_3_v7x1_byte64 = zext i32 %acc_3_v7x1_byte to i64
  %acc_3_v7x1_byte64o = add i64 %acc_3_v7x1_byte64, 2048
  %acc_3_v7x1_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %acc_3_v7x1_byte64o
  %acc_3_v7x1_typed = bitcast i8 addrspace(3)* %acc_3_v7x1_ptr to <2 x half> addrspace(3)*
  %acc_3_v7x1_load = load <2 x half>, <2 x half> addrspace(3)* %acc_3_v7x1_typed, align 4
  %acc_3_v7x1_v2 = bitcast <2 x half> %acc_3_v7x1_load to <2 x half>
  %acc_3_v7x1_sram_e0 = extractelement <2 x half> %acc_3_v7x1_v2, i32 0
  %acc_3_v7x1_sram_e1 = extractelement <2 x half> %acc_3_v7x1_v2, i32 1
  %acc_3_v7x1_sram_v0 = insertelement <64 x half> undef, half %acc_3_v7x1_sram_e0, i32 0
  %acc_3_v7x1_sram = insertelement <64 x half> %acc_3_v7x1_sram_v0, half %acc_3_v7x1_sram_e1, i32 1
  %acc_3_c7d1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %sp_p_7, <64 x half> %acc_3_v7x1_sram, <64 x float> %acc_3_c6d1) #3
  call void @air.wg.barrier(i32 2, i32 1)

  br label %acc_after_head
acc_after_head:
  %acc_o_final_0 = bitcast <64 x float> %acc_0_c7d0 to <64 x float>
  %acc_o_final_1 = bitcast <64 x float> %acc_0_c7d1 to <64 x float>
  %acc_o_final_2 = bitcast <64 x float> %acc_1_c7d0 to <64 x float>
  %acc_o_final_3 = bitcast <64 x float> %acc_1_c7d1 to <64 x float>
  %acc_o_final_4 = bitcast <64 x float> %acc_2_c7d0 to <64 x float>
  %acc_o_final_5 = bitcast <64 x float> %acc_2_c7d1 to <64 x float>
  %acc_o_final_6 = bitcast <64 x float> %acc_3_c7d0 to <64 x float>
  %acc_o_final_7 = bitcast <64 x float> %acc_3_c7d1 to <64 x float>

  ; === Prefetch K[c+blockT, d_outer=0] → TG slot A ===
  %c_next = add i32 %c, 64
  ; === Sync copy (pk_) — all threads cooperative ===
  %pk_sc_t0 = mul i32 %sidx, 32
  %pk_sc_tid = add i32 %pk_sc_t0, %lane_id
  %pk_sc_drem = sub i32 64, 0
  %pk_sc_dcmp = icmp ult i32 %pk_sc_drem, 16
  %pk_sc_dsrc = select i1 %pk_sc_dcmp, i32 %pk_sc_drem, i32 16
  %pk_sc_soob = icmp uge i32 %c_next, 64
  %pk_sc_srr = sub i32 64, %c_next
  %pk_sc_srem = select i1 %pk_sc_soob, i32 0, i32 %pk_sc_srr
  %pk_sc_scmp = icmp ult i32 %pk_sc_srem, 64
  %pk_sc_ssrc = select i1 %pk_sc_scmp, i32 %pk_sc_srem, i32 64
  br label %pk_sc_pre

pk_sc_pre:
  br label %pk_sc_hdr

pk_sc_hdr:
  %pk_sc_i = phi i32 [%pk_sc_tid, %pk_sc_pre], [%pk_sc_inx, %pk_sc_st]
  %pk_sc_done = icmp uge i32 %pk_sc_i, 1024
  br i1 %pk_sc_done, label %pk_sc_end, label %pk_sc_body

pk_sc_body:
  %pk_sc_row = lshr i32 %pk_sc_i, 4
  %pk_sc_col = and i32 %pk_sc_i, 15
  %pk_sc_rok = icmp ult i32 %pk_sc_row, %pk_sc_ssrc
  %pk_sc_cok = icmp ult i32 %pk_sc_col, %pk_sc_dsrc
  %pk_sc_ib = and i1 %pk_sc_rok, %pk_sc_cok
  br i1 %pk_sc_ib, label %pk_sc_ld, label %pk_sc_zr

pk_sc_ld:
  %pk_sc_sr = add i32 %c_next, %pk_sc_row
  %pk_sc_sa = mul i32 %pk_sc_sr, 64
  %pk_sc_sc = add i32 0, %pk_sc_col
  %pk_sc_sad = add i32 %pk_sc_sa, %pk_sc_sc
  %pk_sc_soff = zext i32 %pk_sc_sad to i64
  %pk_sc_sbyt = mul i64 %pk_sc_soff, 2
  %pk_sc_sp = getelementptr i8, i8 addrspace(1)* %K, i64 %pk_sc_sbyt
  %pk_sc_spt = bitcast i8 addrspace(1)* %pk_sc_sp to i16 addrspace(1)*
  %pk_sc_lv = load i16, i16 addrspace(1)* %pk_sc_spt
  br label %pk_sc_st

pk_sc_zr:
  br label %pk_sc_st

pk_sc_st:
  %pk_sc_val = phi i16 [%pk_sc_lv, %pk_sc_ld], [0, %pk_sc_zr]
  %pk_sc_tr = mul i32 %pk_sc_row, 16
  %pk_sc_ta = add i32 %pk_sc_tr, %pk_sc_col
  %pk_sc_tb = mul i32 %pk_sc_ta, 2
  %pk_sc_tb64 = zext i32 %pk_sc_tb to i64
  %pk_sc_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %pk_sc_tb64
  %pk_sc_tpt = bitcast i8 addrspace(3)* %pk_sc_tp to i16 addrspace(3)*
  store i16 %pk_sc_val, i16 addrspace(3)* %pk_sc_tpt
  %pk_sc_inx = add i32 %pk_sc_i, 64
  br label %pk_sc_hdr

pk_sc_end:
  call void @air.wg.barrier(i32 2, i32 1)

  ; Sync wait (wk_) — barrier only (data already in TG)
  call void @air.wg.barrier(i32 2, i32 1)
  br label %wk_after_wait

wk_after_wait:
  br label %loop_latch

loop_latch:
  %m_updated = phi float [%co_m_upd, %wk_after_wait]
  %l_updated = phi float [%rs_l_new, %wk_after_wait]
  %o_acc_0 = phi <64 x float> [%acc_o_final_0, %wk_after_wait]
  %o_acc_1 = phi <64 x float> [%acc_o_final_1, %wk_after_wait]
  %o_acc_2 = phi <64 x float> [%acc_o_final_2, %wk_after_wait]
  %o_acc_3 = phi <64 x float> [%acc_o_final_3, %wk_after_wait]
  %o_acc_4 = phi <64 x float> [%acc_o_final_4, %wk_after_wait]
  %o_acc_5 = phi <64 x float> [%acc_o_final_5, %wk_after_wait]
  %o_acc_6 = phi <64 x float> [%acc_o_final_6, %wk_after_wait]
  %o_acc_7 = phi <64 x float> [%acc_o_final_7, %wk_after_wait]

  br label %loop_header

cleanup:
  ; === Forward cleanup: O /= l, store O, store L ===
  %cl_inv_l = fdiv fast float 1.0, %l_phi
  %cl_oe0_0 = extractelement <64 x float> %o_phi_0, i32 0
  %cl_oe1_0 = extractelement <64 x float> %o_phi_0, i32 1
  %cl_os0_0 = fmul fast float %cl_oe0_0, %cl_inv_l
  %cl_os1_0 = fmul fast float %cl_oe1_0, %cl_inv_l
  %cl_ov0_0 = insertelement <64 x float> %o_phi_0, float %cl_os0_0, i32 0
  %cl_o_scaled_0 = insertelement <64 x float> %cl_ov0_0, float %cl_os1_0, i32 1
  %cl_oe0_1 = extractelement <64 x float> %o_phi_1, i32 0
  %cl_oe1_1 = extractelement <64 x float> %o_phi_1, i32 1
  %cl_os0_1 = fmul fast float %cl_oe0_1, %cl_inv_l
  %cl_os1_1 = fmul fast float %cl_oe1_1, %cl_inv_l
  %cl_ov0_1 = insertelement <64 x float> %o_phi_1, float %cl_os0_1, i32 0
  %cl_o_scaled_1 = insertelement <64 x float> %cl_ov0_1, float %cl_os1_1, i32 1
  %cl_oe0_2 = extractelement <64 x float> %o_phi_2, i32 0
  %cl_oe1_2 = extractelement <64 x float> %o_phi_2, i32 1
  %cl_os0_2 = fmul fast float %cl_oe0_2, %cl_inv_l
  %cl_os1_2 = fmul fast float %cl_oe1_2, %cl_inv_l
  %cl_ov0_2 = insertelement <64 x float> %o_phi_2, float %cl_os0_2, i32 0
  %cl_o_scaled_2 = insertelement <64 x float> %cl_ov0_2, float %cl_os1_2, i32 1
  %cl_oe0_3 = extractelement <64 x float> %o_phi_3, i32 0
  %cl_oe1_3 = extractelement <64 x float> %o_phi_3, i32 1
  %cl_os0_3 = fmul fast float %cl_oe0_3, %cl_inv_l
  %cl_os1_3 = fmul fast float %cl_oe1_3, %cl_inv_l
  %cl_ov0_3 = insertelement <64 x float> %o_phi_3, float %cl_os0_3, i32 0
  %cl_o_scaled_3 = insertelement <64 x float> %cl_ov0_3, float %cl_os1_3, i32 1
  %cl_oe0_4 = extractelement <64 x float> %o_phi_4, i32 0
  %cl_oe1_4 = extractelement <64 x float> %o_phi_4, i32 1
  %cl_os0_4 = fmul fast float %cl_oe0_4, %cl_inv_l
  %cl_os1_4 = fmul fast float %cl_oe1_4, %cl_inv_l
  %cl_ov0_4 = insertelement <64 x float> %o_phi_4, float %cl_os0_4, i32 0
  %cl_o_scaled_4 = insertelement <64 x float> %cl_ov0_4, float %cl_os1_4, i32 1
  %cl_oe0_5 = extractelement <64 x float> %o_phi_5, i32 0
  %cl_oe1_5 = extractelement <64 x float> %o_phi_5, i32 1
  %cl_os0_5 = fmul fast float %cl_oe0_5, %cl_inv_l
  %cl_os1_5 = fmul fast float %cl_oe1_5, %cl_inv_l
  %cl_ov0_5 = insertelement <64 x float> %o_phi_5, float %cl_os0_5, i32 0
  %cl_o_scaled_5 = insertelement <64 x float> %cl_ov0_5, float %cl_os1_5, i32 1
  %cl_oe0_6 = extractelement <64 x float> %o_phi_6, i32 0
  %cl_oe1_6 = extractelement <64 x float> %o_phi_6, i32 1
  %cl_os0_6 = fmul fast float %cl_oe0_6, %cl_inv_l
  %cl_os1_6 = fmul fast float %cl_oe1_6, %cl_inv_l
  %cl_ov0_6 = insertelement <64 x float> %o_phi_6, float %cl_os0_6, i32 0
  %cl_o_scaled_6 = insertelement <64 x float> %cl_ov0_6, float %cl_os1_6, i32 1
  %cl_oe0_7 = extractelement <64 x float> %o_phi_7, i32 0
  %cl_oe1_7 = extractelement <64 x float> %o_phi_7, i32 1
  %cl_os0_7 = fmul fast float %cl_oe0_7, %cl_inv_l
  %cl_os1_7 = fmul fast float %cl_oe1_7, %cl_inv_l
  %cl_ov0_7 = insertelement <64 x float> %o_phi_7, float %cl_os0_7, i32 0
  %cl_o_scaled_7 = insertelement <64 x float> %cl_ov0_7, float %cl_os1_7, i32 1
  ; Store O block d_outer=0 to TG
  %cl_0_k0_v2_e0 = extractelement <64 x float> %cl_o_scaled_0, i32 0
  %cl_0_k0_v2_e1 = extractelement <64 x float> %cl_o_scaled_0, i32 1
  %cl_0_k0_v2_v0 = insertelement <2 x float> undef, float %cl_0_k0_v2_e0, i32 0
  %cl_0_k0_v2 = insertelement <2 x float> %cl_0_k0_v2_v0, float %cl_0_k0_v2_e1, i32 1
  %cl_0_k0_se0 = extractelement <2 x float> %cl_0_k0_v2, i32 0
  %cl_0_k0_se1 = extractelement <2 x float> %cl_0_k0_v2, i32 1
  %cl_0_k0_st0 = fptrunc float %cl_0_k0_se0 to half
  %cl_0_k0_st1 = fptrunc float %cl_0_k0_se1 to half
  %cl_0_k0_sv0 = insertelement <2 x half> undef, half %cl_0_k0_st0, i32 0
  %cl_0_k0_svec = insertelement <2 x half> %cl_0_k0_sv0, half %cl_0_k0_st1, i32 1
  %cl_0_k0_in_bounds = icmp ult i32 %unsafe_par_off, 64
  br i1 %cl_0_k0_in_bounds, label %cl_0_k0_do_store, label %cl_0_k0_skip_store

cl_0_k0_do_store:
  %cl_0_k0_tg_row = add i32 %oig_y, 0
  %cl_0_k0_tg_addr = mul i32 %cl_0_k0_tg_row, 16
  %cl_0_k0_tg_col = add i32 %morton_x, 0
  %cl_0_k0_tg_addr2 = add i32 %cl_0_k0_tg_addr, %cl_0_k0_tg_col
  %cl_0_k0_tg_byte = mul i32 %cl_0_k0_tg_addr2, 2
  %cl_0_k0_tg_byte64 = zext i32 %cl_0_k0_tg_byte to i64
  %cl_0_k0_tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_0_k0_tg_byte64
  %cl_0_k0_tg_typed = bitcast i8 addrspace(3)* %cl_0_k0_tg_ptr to <2 x half> addrspace(3)*
  store <2 x half> %cl_0_k0_svec, <2 x half> addrspace(3)* %cl_0_k0_tg_typed
  br label %cl_0_k0_skip_store

cl_0_k0_skip_store:
  %cl_0_k1_v2_e0 = extractelement <64 x float> %cl_o_scaled_1, i32 0
  %cl_0_k1_v2_e1 = extractelement <64 x float> %cl_o_scaled_1, i32 1
  %cl_0_k1_v2_v0 = insertelement <2 x float> undef, float %cl_0_k1_v2_e0, i32 0
  %cl_0_k1_v2 = insertelement <2 x float> %cl_0_k1_v2_v0, float %cl_0_k1_v2_e1, i32 1
  %cl_0_k1_se0 = extractelement <2 x float> %cl_0_k1_v2, i32 0
  %cl_0_k1_se1 = extractelement <2 x float> %cl_0_k1_v2, i32 1
  %cl_0_k1_st0 = fptrunc float %cl_0_k1_se0 to half
  %cl_0_k1_st1 = fptrunc float %cl_0_k1_se1 to half
  %cl_0_k1_sv0 = insertelement <2 x half> undef, half %cl_0_k1_st0, i32 0
  %cl_0_k1_svec = insertelement <2 x half> %cl_0_k1_sv0, half %cl_0_k1_st1, i32 1
  %cl_0_k1_in_bounds = icmp ult i32 %unsafe_par_off, 64
  br i1 %cl_0_k1_in_bounds, label %cl_0_k1_do_store, label %cl_0_k1_skip_store

cl_0_k1_do_store:
  %cl_0_k1_tg_row = add i32 %oig_y, 0
  %cl_0_k1_tg_addr = mul i32 %cl_0_k1_tg_row, 16
  %cl_0_k1_tg_col = add i32 %morton_x, 8
  %cl_0_k1_tg_addr2 = add i32 %cl_0_k1_tg_addr, %cl_0_k1_tg_col
  %cl_0_k1_tg_byte = mul i32 %cl_0_k1_tg_addr2, 2
  %cl_0_k1_tg_byte64 = zext i32 %cl_0_k1_tg_byte to i64
  %cl_0_k1_tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_0_k1_tg_byte64
  %cl_0_k1_tg_typed = bitcast i8 addrspace(3)* %cl_0_k1_tg_ptr to <2 x half> addrspace(3)*
  store <2 x half> %cl_0_k1_svec, <2 x half> addrspace(3)* %cl_0_k1_tg_typed
  br label %cl_0_k1_skip_store

cl_0_k1_skip_store:
  call void @air.wg.barrier(i32 2, i32 1)
  %cl_0_cp_dev_row = mul i32 %par_group_off, 64
  %cl_0_cp_dev_off32 = add i32 %cl_0_cp_dev_row, 0
  %cl_0_cp_dev_off = zext i32 %cl_0_cp_dev_off32 to i64
  %cl_0_cp_dev_byte = mul i64 %cl_0_cp_dev_off, 2
  %cl_0_cp_dst_p = getelementptr i8, i8 addrspace(1)* %O, i64 %cl_0_cp_dev_byte
  %cl_0_cp_seq_rem = sub i32 64, %par_group_off
  %cl_0_cp_seq_cmp = icmp ult i32 %cl_0_cp_seq_rem, 16
  %cl_0_cp_seq_tile32 = select i1 %cl_0_cp_seq_cmp, i32 %cl_0_cp_seq_rem, i32 16
  ; === Sync store TG→device (cl_0_cp_) all-threads ===
  %cl_0_cp_ss_t0 = mul i32 %sidx, 32
  %cl_0_cp_ss_tid = add i32 %cl_0_cp_ss_t0, %lane_id
  br label %cl_0_cp_ss_pre

cl_0_cp_ss_pre:
  br label %cl_0_cp_ss_hdr

cl_0_cp_ss_hdr:
  %cl_0_cp_ss_i = phi i32 [%cl_0_cp_ss_tid, %cl_0_cp_ss_pre], [%cl_0_cp_ss_inx, %cl_0_cp_ss_nx]
  %cl_0_cp_ss_done = icmp uge i32 %cl_0_cp_ss_i, 256
  br i1 %cl_0_cp_ss_done, label %cl_0_cp_ss_end, label %cl_0_cp_ss_body

cl_0_cp_ss_body:
  %cl_0_cp_ss_row = lshr i32 %cl_0_cp_ss_i, 4
  %cl_0_cp_ss_col = and i32 %cl_0_cp_ss_i, 15
  %cl_0_cp_ss_rok = icmp ult i32 %cl_0_cp_ss_row, %cl_0_cp_seq_tile32
  %cl_0_cp_ss_cok = icmp ult i32 %cl_0_cp_ss_col, 16
  %cl_0_cp_ss_ib = and i1 %cl_0_cp_ss_rok, %cl_0_cp_ss_cok
  br i1 %cl_0_cp_ss_ib, label %cl_0_cp_ss_do, label %cl_0_cp_ss_nx

cl_0_cp_ss_do:
  %cl_0_cp_ss_tr = mul i32 %cl_0_cp_ss_row, 16
  %cl_0_cp_ss_ta = add i32 %cl_0_cp_ss_tr, %cl_0_cp_ss_col
  %cl_0_cp_ss_tb = mul i32 %cl_0_cp_ss_ta, 2
  %cl_0_cp_ss_tb64 = zext i32 %cl_0_cp_ss_tb to i64
  %cl_0_cp_ss_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_0_cp_ss_tb64
  %cl_0_cp_ss_tpt = bitcast i8 addrspace(3)* %cl_0_cp_ss_tp to i16 addrspace(3)*
  %cl_0_cp_ss_lv = load i16, i16 addrspace(3)* %cl_0_cp_ss_tpt
  %cl_0_cp_ss_dr = mul i32 %cl_0_cp_ss_row, 64
  %cl_0_cp_ss_da = add i32 %cl_0_cp_ss_dr, %cl_0_cp_ss_col
  %cl_0_cp_ss_db = mul i32 %cl_0_cp_ss_da, 2
  %cl_0_cp_ss_db64 = zext i32 %cl_0_cp_ss_db to i64
  %cl_0_cp_ss_dp = getelementptr i8, i8 addrspace(1)* %cl_0_cp_dst_p, i64 %cl_0_cp_ss_db64
  %cl_0_cp_ss_dpt = bitcast i8 addrspace(1)* %cl_0_cp_ss_dp to i16 addrspace(1)*
  store i16 %cl_0_cp_ss_lv, i16 addrspace(1)* %cl_0_cp_ss_dpt
  br label %cl_0_cp_ss_nx

cl_0_cp_ss_nx:
  %cl_0_cp_ss_inx = add i32 %cl_0_cp_ss_i, 64
  br label %cl_0_cp_ss_hdr

cl_0_cp_ss_end:
  call void @air.wg.barrier(i32 2, i32 1)

  ; Store O block d_outer=16 to TG
  %cl_1_k0_v2_e0 = extractelement <64 x float> %cl_o_scaled_2, i32 0
  %cl_1_k0_v2_e1 = extractelement <64 x float> %cl_o_scaled_2, i32 1
  %cl_1_k0_v2_v0 = insertelement <2 x float> undef, float %cl_1_k0_v2_e0, i32 0
  %cl_1_k0_v2 = insertelement <2 x float> %cl_1_k0_v2_v0, float %cl_1_k0_v2_e1, i32 1
  %cl_1_k0_se0 = extractelement <2 x float> %cl_1_k0_v2, i32 0
  %cl_1_k0_se1 = extractelement <2 x float> %cl_1_k0_v2, i32 1
  %cl_1_k0_st0 = fptrunc float %cl_1_k0_se0 to half
  %cl_1_k0_st1 = fptrunc float %cl_1_k0_se1 to half
  %cl_1_k0_sv0 = insertelement <2 x half> undef, half %cl_1_k0_st0, i32 0
  %cl_1_k0_svec = insertelement <2 x half> %cl_1_k0_sv0, half %cl_1_k0_st1, i32 1
  %cl_1_k0_in_bounds = icmp ult i32 %unsafe_par_off, 64
  br i1 %cl_1_k0_in_bounds, label %cl_1_k0_do_store, label %cl_1_k0_skip_store

cl_1_k0_do_store:
  %cl_1_k0_tg_row = add i32 %oig_y, 0
  %cl_1_k0_tg_addr = mul i32 %cl_1_k0_tg_row, 16
  %cl_1_k0_tg_col = add i32 %morton_x, 0
  %cl_1_k0_tg_addr2 = add i32 %cl_1_k0_tg_addr, %cl_1_k0_tg_col
  %cl_1_k0_tg_byte = mul i32 %cl_1_k0_tg_addr2, 2
  %cl_1_k0_tg_byte64 = zext i32 %cl_1_k0_tg_byte to i64
  %cl_1_k0_tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_1_k0_tg_byte64
  %cl_1_k0_tg_typed = bitcast i8 addrspace(3)* %cl_1_k0_tg_ptr to <2 x half> addrspace(3)*
  store <2 x half> %cl_1_k0_svec, <2 x half> addrspace(3)* %cl_1_k0_tg_typed
  br label %cl_1_k0_skip_store

cl_1_k0_skip_store:
  %cl_1_k1_v2_e0 = extractelement <64 x float> %cl_o_scaled_3, i32 0
  %cl_1_k1_v2_e1 = extractelement <64 x float> %cl_o_scaled_3, i32 1
  %cl_1_k1_v2_v0 = insertelement <2 x float> undef, float %cl_1_k1_v2_e0, i32 0
  %cl_1_k1_v2 = insertelement <2 x float> %cl_1_k1_v2_v0, float %cl_1_k1_v2_e1, i32 1
  %cl_1_k1_se0 = extractelement <2 x float> %cl_1_k1_v2, i32 0
  %cl_1_k1_se1 = extractelement <2 x float> %cl_1_k1_v2, i32 1
  %cl_1_k1_st0 = fptrunc float %cl_1_k1_se0 to half
  %cl_1_k1_st1 = fptrunc float %cl_1_k1_se1 to half
  %cl_1_k1_sv0 = insertelement <2 x half> undef, half %cl_1_k1_st0, i32 0
  %cl_1_k1_svec = insertelement <2 x half> %cl_1_k1_sv0, half %cl_1_k1_st1, i32 1
  %cl_1_k1_in_bounds = icmp ult i32 %unsafe_par_off, 64
  br i1 %cl_1_k1_in_bounds, label %cl_1_k1_do_store, label %cl_1_k1_skip_store

cl_1_k1_do_store:
  %cl_1_k1_tg_row = add i32 %oig_y, 0
  %cl_1_k1_tg_addr = mul i32 %cl_1_k1_tg_row, 16
  %cl_1_k1_tg_col = add i32 %morton_x, 8
  %cl_1_k1_tg_addr2 = add i32 %cl_1_k1_tg_addr, %cl_1_k1_tg_col
  %cl_1_k1_tg_byte = mul i32 %cl_1_k1_tg_addr2, 2
  %cl_1_k1_tg_byte64 = zext i32 %cl_1_k1_tg_byte to i64
  %cl_1_k1_tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_1_k1_tg_byte64
  %cl_1_k1_tg_typed = bitcast i8 addrspace(3)* %cl_1_k1_tg_ptr to <2 x half> addrspace(3)*
  store <2 x half> %cl_1_k1_svec, <2 x half> addrspace(3)* %cl_1_k1_tg_typed
  br label %cl_1_k1_skip_store

cl_1_k1_skip_store:
  call void @air.wg.barrier(i32 2, i32 1)
  %cl_1_cp_dev_row = mul i32 %par_group_off, 64
  %cl_1_cp_dev_off32 = add i32 %cl_1_cp_dev_row, 16
  %cl_1_cp_dev_off = zext i32 %cl_1_cp_dev_off32 to i64
  %cl_1_cp_dev_byte = mul i64 %cl_1_cp_dev_off, 2
  %cl_1_cp_dst_p = getelementptr i8, i8 addrspace(1)* %O, i64 %cl_1_cp_dev_byte
  %cl_1_cp_seq_rem = sub i32 64, %par_group_off
  %cl_1_cp_seq_cmp = icmp ult i32 %cl_1_cp_seq_rem, 16
  %cl_1_cp_seq_tile32 = select i1 %cl_1_cp_seq_cmp, i32 %cl_1_cp_seq_rem, i32 16
  ; === Sync store TG→device (cl_1_cp_) all-threads ===
  %cl_1_cp_ss_t0 = mul i32 %sidx, 32
  %cl_1_cp_ss_tid = add i32 %cl_1_cp_ss_t0, %lane_id
  br label %cl_1_cp_ss_pre

cl_1_cp_ss_pre:
  br label %cl_1_cp_ss_hdr

cl_1_cp_ss_hdr:
  %cl_1_cp_ss_i = phi i32 [%cl_1_cp_ss_tid, %cl_1_cp_ss_pre], [%cl_1_cp_ss_inx, %cl_1_cp_ss_nx]
  %cl_1_cp_ss_done = icmp uge i32 %cl_1_cp_ss_i, 256
  br i1 %cl_1_cp_ss_done, label %cl_1_cp_ss_end, label %cl_1_cp_ss_body

cl_1_cp_ss_body:
  %cl_1_cp_ss_row = lshr i32 %cl_1_cp_ss_i, 4
  %cl_1_cp_ss_col = and i32 %cl_1_cp_ss_i, 15
  %cl_1_cp_ss_rok = icmp ult i32 %cl_1_cp_ss_row, %cl_1_cp_seq_tile32
  %cl_1_cp_ss_cok = icmp ult i32 %cl_1_cp_ss_col, 16
  %cl_1_cp_ss_ib = and i1 %cl_1_cp_ss_rok, %cl_1_cp_ss_cok
  br i1 %cl_1_cp_ss_ib, label %cl_1_cp_ss_do, label %cl_1_cp_ss_nx

cl_1_cp_ss_do:
  %cl_1_cp_ss_tr = mul i32 %cl_1_cp_ss_row, 16
  %cl_1_cp_ss_ta = add i32 %cl_1_cp_ss_tr, %cl_1_cp_ss_col
  %cl_1_cp_ss_tb = mul i32 %cl_1_cp_ss_ta, 2
  %cl_1_cp_ss_tb64 = zext i32 %cl_1_cp_ss_tb to i64
  %cl_1_cp_ss_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_1_cp_ss_tb64
  %cl_1_cp_ss_tpt = bitcast i8 addrspace(3)* %cl_1_cp_ss_tp to i16 addrspace(3)*
  %cl_1_cp_ss_lv = load i16, i16 addrspace(3)* %cl_1_cp_ss_tpt
  %cl_1_cp_ss_dr = mul i32 %cl_1_cp_ss_row, 64
  %cl_1_cp_ss_da = add i32 %cl_1_cp_ss_dr, %cl_1_cp_ss_col
  %cl_1_cp_ss_db = mul i32 %cl_1_cp_ss_da, 2
  %cl_1_cp_ss_db64 = zext i32 %cl_1_cp_ss_db to i64
  %cl_1_cp_ss_dp = getelementptr i8, i8 addrspace(1)* %cl_1_cp_dst_p, i64 %cl_1_cp_ss_db64
  %cl_1_cp_ss_dpt = bitcast i8 addrspace(1)* %cl_1_cp_ss_dp to i16 addrspace(1)*
  store i16 %cl_1_cp_ss_lv, i16 addrspace(1)* %cl_1_cp_ss_dpt
  br label %cl_1_cp_ss_nx

cl_1_cp_ss_nx:
  %cl_1_cp_ss_inx = add i32 %cl_1_cp_ss_i, 64
  br label %cl_1_cp_ss_hdr

cl_1_cp_ss_end:
  call void @air.wg.barrier(i32 2, i32 1)

  ; Store O block d_outer=32 to TG
  %cl_2_k0_v2_e0 = extractelement <64 x float> %cl_o_scaled_4, i32 0
  %cl_2_k0_v2_e1 = extractelement <64 x float> %cl_o_scaled_4, i32 1
  %cl_2_k0_v2_v0 = insertelement <2 x float> undef, float %cl_2_k0_v2_e0, i32 0
  %cl_2_k0_v2 = insertelement <2 x float> %cl_2_k0_v2_v0, float %cl_2_k0_v2_e1, i32 1
  %cl_2_k0_se0 = extractelement <2 x float> %cl_2_k0_v2, i32 0
  %cl_2_k0_se1 = extractelement <2 x float> %cl_2_k0_v2, i32 1
  %cl_2_k0_st0 = fptrunc float %cl_2_k0_se0 to half
  %cl_2_k0_st1 = fptrunc float %cl_2_k0_se1 to half
  %cl_2_k0_sv0 = insertelement <2 x half> undef, half %cl_2_k0_st0, i32 0
  %cl_2_k0_svec = insertelement <2 x half> %cl_2_k0_sv0, half %cl_2_k0_st1, i32 1
  %cl_2_k0_in_bounds = icmp ult i32 %unsafe_par_off, 64
  br i1 %cl_2_k0_in_bounds, label %cl_2_k0_do_store, label %cl_2_k0_skip_store

cl_2_k0_do_store:
  %cl_2_k0_tg_row = add i32 %oig_y, 0
  %cl_2_k0_tg_addr = mul i32 %cl_2_k0_tg_row, 16
  %cl_2_k0_tg_col = add i32 %morton_x, 0
  %cl_2_k0_tg_addr2 = add i32 %cl_2_k0_tg_addr, %cl_2_k0_tg_col
  %cl_2_k0_tg_byte = mul i32 %cl_2_k0_tg_addr2, 2
  %cl_2_k0_tg_byte64 = zext i32 %cl_2_k0_tg_byte to i64
  %cl_2_k0_tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_2_k0_tg_byte64
  %cl_2_k0_tg_typed = bitcast i8 addrspace(3)* %cl_2_k0_tg_ptr to <2 x half> addrspace(3)*
  store <2 x half> %cl_2_k0_svec, <2 x half> addrspace(3)* %cl_2_k0_tg_typed
  br label %cl_2_k0_skip_store

cl_2_k0_skip_store:
  %cl_2_k1_v2_e0 = extractelement <64 x float> %cl_o_scaled_5, i32 0
  %cl_2_k1_v2_e1 = extractelement <64 x float> %cl_o_scaled_5, i32 1
  %cl_2_k1_v2_v0 = insertelement <2 x float> undef, float %cl_2_k1_v2_e0, i32 0
  %cl_2_k1_v2 = insertelement <2 x float> %cl_2_k1_v2_v0, float %cl_2_k1_v2_e1, i32 1
  %cl_2_k1_se0 = extractelement <2 x float> %cl_2_k1_v2, i32 0
  %cl_2_k1_se1 = extractelement <2 x float> %cl_2_k1_v2, i32 1
  %cl_2_k1_st0 = fptrunc float %cl_2_k1_se0 to half
  %cl_2_k1_st1 = fptrunc float %cl_2_k1_se1 to half
  %cl_2_k1_sv0 = insertelement <2 x half> undef, half %cl_2_k1_st0, i32 0
  %cl_2_k1_svec = insertelement <2 x half> %cl_2_k1_sv0, half %cl_2_k1_st1, i32 1
  %cl_2_k1_in_bounds = icmp ult i32 %unsafe_par_off, 64
  br i1 %cl_2_k1_in_bounds, label %cl_2_k1_do_store, label %cl_2_k1_skip_store

cl_2_k1_do_store:
  %cl_2_k1_tg_row = add i32 %oig_y, 0
  %cl_2_k1_tg_addr = mul i32 %cl_2_k1_tg_row, 16
  %cl_2_k1_tg_col = add i32 %morton_x, 8
  %cl_2_k1_tg_addr2 = add i32 %cl_2_k1_tg_addr, %cl_2_k1_tg_col
  %cl_2_k1_tg_byte = mul i32 %cl_2_k1_tg_addr2, 2
  %cl_2_k1_tg_byte64 = zext i32 %cl_2_k1_tg_byte to i64
  %cl_2_k1_tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_2_k1_tg_byte64
  %cl_2_k1_tg_typed = bitcast i8 addrspace(3)* %cl_2_k1_tg_ptr to <2 x half> addrspace(3)*
  store <2 x half> %cl_2_k1_svec, <2 x half> addrspace(3)* %cl_2_k1_tg_typed
  br label %cl_2_k1_skip_store

cl_2_k1_skip_store:
  call void @air.wg.barrier(i32 2, i32 1)
  %cl_2_cp_dev_row = mul i32 %par_group_off, 64
  %cl_2_cp_dev_off32 = add i32 %cl_2_cp_dev_row, 32
  %cl_2_cp_dev_off = zext i32 %cl_2_cp_dev_off32 to i64
  %cl_2_cp_dev_byte = mul i64 %cl_2_cp_dev_off, 2
  %cl_2_cp_dst_p = getelementptr i8, i8 addrspace(1)* %O, i64 %cl_2_cp_dev_byte
  %cl_2_cp_seq_rem = sub i32 64, %par_group_off
  %cl_2_cp_seq_cmp = icmp ult i32 %cl_2_cp_seq_rem, 16
  %cl_2_cp_seq_tile32 = select i1 %cl_2_cp_seq_cmp, i32 %cl_2_cp_seq_rem, i32 16
  ; === Sync store TG→device (cl_2_cp_) all-threads ===
  %cl_2_cp_ss_t0 = mul i32 %sidx, 32
  %cl_2_cp_ss_tid = add i32 %cl_2_cp_ss_t0, %lane_id
  br label %cl_2_cp_ss_pre

cl_2_cp_ss_pre:
  br label %cl_2_cp_ss_hdr

cl_2_cp_ss_hdr:
  %cl_2_cp_ss_i = phi i32 [%cl_2_cp_ss_tid, %cl_2_cp_ss_pre], [%cl_2_cp_ss_inx, %cl_2_cp_ss_nx]
  %cl_2_cp_ss_done = icmp uge i32 %cl_2_cp_ss_i, 256
  br i1 %cl_2_cp_ss_done, label %cl_2_cp_ss_end, label %cl_2_cp_ss_body

cl_2_cp_ss_body:
  %cl_2_cp_ss_row = lshr i32 %cl_2_cp_ss_i, 4
  %cl_2_cp_ss_col = and i32 %cl_2_cp_ss_i, 15
  %cl_2_cp_ss_rok = icmp ult i32 %cl_2_cp_ss_row, %cl_2_cp_seq_tile32
  %cl_2_cp_ss_cok = icmp ult i32 %cl_2_cp_ss_col, 16
  %cl_2_cp_ss_ib = and i1 %cl_2_cp_ss_rok, %cl_2_cp_ss_cok
  br i1 %cl_2_cp_ss_ib, label %cl_2_cp_ss_do, label %cl_2_cp_ss_nx

cl_2_cp_ss_do:
  %cl_2_cp_ss_tr = mul i32 %cl_2_cp_ss_row, 16
  %cl_2_cp_ss_ta = add i32 %cl_2_cp_ss_tr, %cl_2_cp_ss_col
  %cl_2_cp_ss_tb = mul i32 %cl_2_cp_ss_ta, 2
  %cl_2_cp_ss_tb64 = zext i32 %cl_2_cp_ss_tb to i64
  %cl_2_cp_ss_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_2_cp_ss_tb64
  %cl_2_cp_ss_tpt = bitcast i8 addrspace(3)* %cl_2_cp_ss_tp to i16 addrspace(3)*
  %cl_2_cp_ss_lv = load i16, i16 addrspace(3)* %cl_2_cp_ss_tpt
  %cl_2_cp_ss_dr = mul i32 %cl_2_cp_ss_row, 64
  %cl_2_cp_ss_da = add i32 %cl_2_cp_ss_dr, %cl_2_cp_ss_col
  %cl_2_cp_ss_db = mul i32 %cl_2_cp_ss_da, 2
  %cl_2_cp_ss_db64 = zext i32 %cl_2_cp_ss_db to i64
  %cl_2_cp_ss_dp = getelementptr i8, i8 addrspace(1)* %cl_2_cp_dst_p, i64 %cl_2_cp_ss_db64
  %cl_2_cp_ss_dpt = bitcast i8 addrspace(1)* %cl_2_cp_ss_dp to i16 addrspace(1)*
  store i16 %cl_2_cp_ss_lv, i16 addrspace(1)* %cl_2_cp_ss_dpt
  br label %cl_2_cp_ss_nx

cl_2_cp_ss_nx:
  %cl_2_cp_ss_inx = add i32 %cl_2_cp_ss_i, 64
  br label %cl_2_cp_ss_hdr

cl_2_cp_ss_end:
  call void @air.wg.barrier(i32 2, i32 1)

  ; Store O block d_outer=48 to TG
  %cl_3_k0_v2_e0 = extractelement <64 x float> %cl_o_scaled_6, i32 0
  %cl_3_k0_v2_e1 = extractelement <64 x float> %cl_o_scaled_6, i32 1
  %cl_3_k0_v2_v0 = insertelement <2 x float> undef, float %cl_3_k0_v2_e0, i32 0
  %cl_3_k0_v2 = insertelement <2 x float> %cl_3_k0_v2_v0, float %cl_3_k0_v2_e1, i32 1
  %cl_3_k0_se0 = extractelement <2 x float> %cl_3_k0_v2, i32 0
  %cl_3_k0_se1 = extractelement <2 x float> %cl_3_k0_v2, i32 1
  %cl_3_k0_st0 = fptrunc float %cl_3_k0_se0 to half
  %cl_3_k0_st1 = fptrunc float %cl_3_k0_se1 to half
  %cl_3_k0_sv0 = insertelement <2 x half> undef, half %cl_3_k0_st0, i32 0
  %cl_3_k0_svec = insertelement <2 x half> %cl_3_k0_sv0, half %cl_3_k0_st1, i32 1
  %cl_3_k0_in_bounds = icmp ult i32 %unsafe_par_off, 64
  br i1 %cl_3_k0_in_bounds, label %cl_3_k0_do_store, label %cl_3_k0_skip_store

cl_3_k0_do_store:
  %cl_3_k0_tg_row = add i32 %oig_y, 0
  %cl_3_k0_tg_addr = mul i32 %cl_3_k0_tg_row, 16
  %cl_3_k0_tg_col = add i32 %morton_x, 0
  %cl_3_k0_tg_addr2 = add i32 %cl_3_k0_tg_addr, %cl_3_k0_tg_col
  %cl_3_k0_tg_byte = mul i32 %cl_3_k0_tg_addr2, 2
  %cl_3_k0_tg_byte64 = zext i32 %cl_3_k0_tg_byte to i64
  %cl_3_k0_tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_3_k0_tg_byte64
  %cl_3_k0_tg_typed = bitcast i8 addrspace(3)* %cl_3_k0_tg_ptr to <2 x half> addrspace(3)*
  store <2 x half> %cl_3_k0_svec, <2 x half> addrspace(3)* %cl_3_k0_tg_typed
  br label %cl_3_k0_skip_store

cl_3_k0_skip_store:
  %cl_3_k1_v2_e0 = extractelement <64 x float> %cl_o_scaled_7, i32 0
  %cl_3_k1_v2_e1 = extractelement <64 x float> %cl_o_scaled_7, i32 1
  %cl_3_k1_v2_v0 = insertelement <2 x float> undef, float %cl_3_k1_v2_e0, i32 0
  %cl_3_k1_v2 = insertelement <2 x float> %cl_3_k1_v2_v0, float %cl_3_k1_v2_e1, i32 1
  %cl_3_k1_se0 = extractelement <2 x float> %cl_3_k1_v2, i32 0
  %cl_3_k1_se1 = extractelement <2 x float> %cl_3_k1_v2, i32 1
  %cl_3_k1_st0 = fptrunc float %cl_3_k1_se0 to half
  %cl_3_k1_st1 = fptrunc float %cl_3_k1_se1 to half
  %cl_3_k1_sv0 = insertelement <2 x half> undef, half %cl_3_k1_st0, i32 0
  %cl_3_k1_svec = insertelement <2 x half> %cl_3_k1_sv0, half %cl_3_k1_st1, i32 1
  %cl_3_k1_in_bounds = icmp ult i32 %unsafe_par_off, 64
  br i1 %cl_3_k1_in_bounds, label %cl_3_k1_do_store, label %cl_3_k1_skip_store

cl_3_k1_do_store:
  %cl_3_k1_tg_row = add i32 %oig_y, 0
  %cl_3_k1_tg_addr = mul i32 %cl_3_k1_tg_row, 16
  %cl_3_k1_tg_col = add i32 %morton_x, 8
  %cl_3_k1_tg_addr2 = add i32 %cl_3_k1_tg_addr, %cl_3_k1_tg_col
  %cl_3_k1_tg_byte = mul i32 %cl_3_k1_tg_addr2, 2
  %cl_3_k1_tg_byte64 = zext i32 %cl_3_k1_tg_byte to i64
  %cl_3_k1_tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_3_k1_tg_byte64
  %cl_3_k1_tg_typed = bitcast i8 addrspace(3)* %cl_3_k1_tg_ptr to <2 x half> addrspace(3)*
  store <2 x half> %cl_3_k1_svec, <2 x half> addrspace(3)* %cl_3_k1_tg_typed
  br label %cl_3_k1_skip_store

cl_3_k1_skip_store:
  call void @air.wg.barrier(i32 2, i32 1)
  %cl_3_cp_dev_row = mul i32 %par_group_off, 64
  %cl_3_cp_dev_off32 = add i32 %cl_3_cp_dev_row, 48
  %cl_3_cp_dev_off = zext i32 %cl_3_cp_dev_off32 to i64
  %cl_3_cp_dev_byte = mul i64 %cl_3_cp_dev_off, 2
  %cl_3_cp_dst_p = getelementptr i8, i8 addrspace(1)* %O, i64 %cl_3_cp_dev_byte
  %cl_3_cp_seq_rem = sub i32 64, %par_group_off
  %cl_3_cp_seq_cmp = icmp ult i32 %cl_3_cp_seq_rem, 16
  %cl_3_cp_seq_tile32 = select i1 %cl_3_cp_seq_cmp, i32 %cl_3_cp_seq_rem, i32 16
  ; === Sync store TG→device (cl_3_cp_) all-threads ===
  %cl_3_cp_ss_t0 = mul i32 %sidx, 32
  %cl_3_cp_ss_tid = add i32 %cl_3_cp_ss_t0, %lane_id
  br label %cl_3_cp_ss_pre

cl_3_cp_ss_pre:
  br label %cl_3_cp_ss_hdr

cl_3_cp_ss_hdr:
  %cl_3_cp_ss_i = phi i32 [%cl_3_cp_ss_tid, %cl_3_cp_ss_pre], [%cl_3_cp_ss_inx, %cl_3_cp_ss_nx]
  %cl_3_cp_ss_done = icmp uge i32 %cl_3_cp_ss_i, 256
  br i1 %cl_3_cp_ss_done, label %cl_3_cp_ss_end, label %cl_3_cp_ss_body

cl_3_cp_ss_body:
  %cl_3_cp_ss_row = lshr i32 %cl_3_cp_ss_i, 4
  %cl_3_cp_ss_col = and i32 %cl_3_cp_ss_i, 15
  %cl_3_cp_ss_rok = icmp ult i32 %cl_3_cp_ss_row, %cl_3_cp_seq_tile32
  %cl_3_cp_ss_cok = icmp ult i32 %cl_3_cp_ss_col, 16
  %cl_3_cp_ss_ib = and i1 %cl_3_cp_ss_rok, %cl_3_cp_ss_cok
  br i1 %cl_3_cp_ss_ib, label %cl_3_cp_ss_do, label %cl_3_cp_ss_nx

cl_3_cp_ss_do:
  %cl_3_cp_ss_tr = mul i32 %cl_3_cp_ss_row, 16
  %cl_3_cp_ss_ta = add i32 %cl_3_cp_ss_tr, %cl_3_cp_ss_col
  %cl_3_cp_ss_tb = mul i32 %cl_3_cp_ss_ta, 2
  %cl_3_cp_ss_tb64 = zext i32 %cl_3_cp_ss_tb to i64
  %cl_3_cp_ss_tp = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cl_3_cp_ss_tb64
  %cl_3_cp_ss_tpt = bitcast i8 addrspace(3)* %cl_3_cp_ss_tp to i16 addrspace(3)*
  %cl_3_cp_ss_lv = load i16, i16 addrspace(3)* %cl_3_cp_ss_tpt
  %cl_3_cp_ss_dr = mul i32 %cl_3_cp_ss_row, 64
  %cl_3_cp_ss_da = add i32 %cl_3_cp_ss_dr, %cl_3_cp_ss_col
  %cl_3_cp_ss_db = mul i32 %cl_3_cp_ss_da, 2
  %cl_3_cp_ss_db64 = zext i32 %cl_3_cp_ss_db to i64
  %cl_3_cp_ss_dp = getelementptr i8, i8 addrspace(1)* %cl_3_cp_dst_p, i64 %cl_3_cp_ss_db64
  %cl_3_cp_ss_dpt = bitcast i8 addrspace(1)* %cl_3_cp_ss_dp to i16 addrspace(1)*
  store i16 %cl_3_cp_ss_lv, i16 addrspace(1)* %cl_3_cp_ss_dpt
  br label %cl_3_cp_ss_nx

cl_3_cp_ss_nx:
  %cl_3_cp_ss_inx = add i32 %cl_3_cp_ss_i, 64
  br label %cl_3_cp_ss_hdr

cl_3_cp_ss_end:
  call void @air.wg.barrier(i32 2, i32 1)

  ; Store L
  %cl_L_in_bounds = icmp ult i32 %unsafe_par_off, 64
  br i1 %cl_L_in_bounds, label %cl_store_L, label %cl_skip_L

cl_store_L:
  %cl_log2_l = call fast float @llvm.log2.f32(float %l_phi)
  %cl_L_val = fadd fast float %m_phi, %cl_log2_l
  %cl_L_off = zext i32 %clamped_par_off to i64
  %cl_L_byte = mul i64 %cl_L_off, 4
  %cl_L_ptr = getelementptr i8, i8 addrspace(1)* %L_buf, i64 %cl_L_byte
  %cl_L_typed = bitcast i8 addrspace(1)* %cl_L_ptr to float addrspace(1)*
  store float %cl_L_val, float addrspace(1)* %cl_L_typed
  br label %cl_skip_L

cl_skip_L:
  br label %exit

  exit:
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
