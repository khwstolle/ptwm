// {{ name }} — {{ description }}
// PTWM {{ kind }} contribution, ABI v1.

let cursor: u32 = 0;

export function ptwm_alloc(len: u32): u32 {
  const p = cursor;
  cursor += len;
  return p;
}

export function ptwm_free(_ptr: u32, _len: u32): void {}

export function ptwm_{{ kind }}_v1_encode(
  in_ptr: u32, in_len: u32, out_ptr: u32, out_len: u32
): i64 {
  if (out_len < in_len) return -2;
  for (let i: u32 = 0; i < in_len; i++) {
    store<u8>(out_ptr + i, load<u8>(in_ptr + i));
  }
  return i64(in_len);
}

export function ptwm_{{ kind }}_v1_decode(
  in_ptr: u32, in_len: u32, out_ptr: u32, out_len: u32
): i64 {
  return ptwm_{{ kind }}_v1_encode(in_ptr, in_len, out_ptr, out_len);
}
