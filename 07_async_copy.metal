#include <metal_stdlib>
using namespace metal;

kernel void test_copy(
    device float* src [[buffer(0)]],
    threadgroup float* tg [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]]
) {
    // Test if simdgroup_async_copy is a built-in function
    simdgroup_event event;
    event.async_copy(tg, src, 32);
    simdgroup_event::wait(1, &event);
    
    // Alternative: threadgroup_async_copy
    // async event_t e = async_work_group_copy(tg, src, 32, 0);
    // wait_group_events(1, &e);
}
