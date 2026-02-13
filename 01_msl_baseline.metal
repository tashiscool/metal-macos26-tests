#include <metal_stdlib>
using namespace metal;

// Baseline MSL kernel — always works on macOS 26.
// The MSL compiler adds SDK Version and other metadata automatically.
kernel void test_kernel(
    device uchar* Q [[buffer(0)]],
    device uchar* K [[buffer(1)]],
    device uchar* O [[buffer(3)]],
    threadgroup uchar* tg_base [[threadgroup(0)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]]
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset = gid.x * 32 + sidx * 8 + lane_id;
    device float* out = (device float*)(O + offset * 4);
    *out = float(offset);
}
