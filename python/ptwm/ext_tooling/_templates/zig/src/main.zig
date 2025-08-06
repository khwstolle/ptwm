// {{ name }} — {{ description }}
// PTWM {{ kind }} contribution, ABI v1.

const std = @import("std");

var heap: [1 << 20]u8 = undefined;
var cursor: u32 = 0;

export fn ptwm_alloc(len: u32) u32 {
    const p = cursor;
    cursor += len;
    return @intFromPtr(&heap[p]);
}

export fn ptwm_free(ptr: u32, len: u32) void {
    _ = ptr;
    _ = len;
}

export fn ptwm_{{ kind }}_v1_encode(in_ptr: u32, in_len: u32, out_ptr: u32, out_len: u32) i64 {
    if (out_len < in_len) return -2;
    const in_slice = @as([*]const u8, @ptrFromInt(in_ptr))[0..in_len];
    var out_slice = @as([*]u8, @ptrFromInt(out_ptr))[0..out_len];
    @memcpy(out_slice[0..in_len], in_slice);
    return @as(i64, @intCast(in_len));
}

export fn ptwm_{{ kind }}_v1_decode(in_ptr: u32, in_len: u32, out_ptr: u32, out_len: u32) i64 {
    return ptwm_{{ kind }}_v1_encode(in_ptr, in_len, out_ptr, out_len);
}
