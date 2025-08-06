/* {{ name }} — {{ description }}
 * PTWM {{ kind }} contribution, ABI v1.
 */

#include <stddef.h>
#include <stdint.h>

extern void *ptwm_malloc(size_t);
extern void  ptwm_realfree(void *);

uint32_t ptwm_alloc(uint32_t len) {
  /* Minimal bump allocator backed by a static buffer. */
  static unsigned char heap[1 << 20];
  static uint32_t cursor = 0;
  uint32_t p = cursor;
  cursor += len;
  return (uint32_t)(uintptr_t)&heap[p];
}

void ptwm_free(uint32_t ptr, uint32_t len) {
  (void)ptr;
  (void)len;
}

int64_t ptwm_{{ kind }}_v1_encode(
  uint32_t in_ptr, uint32_t in_len, uint32_t out_ptr, uint32_t out_len) {
  if (out_len < in_len) return -2;
  unsigned char *in = (unsigned char *)(uintptr_t)in_ptr;
  unsigned char *out = (unsigned char *)(uintptr_t)out_ptr;
  for (uint32_t i = 0; i < in_len; ++i) out[i] = in[i];
  return (int64_t)in_len;
}

int64_t ptwm_{{ kind }}_v1_decode(
  uint32_t in_ptr, uint32_t in_len, uint32_t out_ptr, uint32_t out_len) {
  return ptwm_{{ kind }}_v1_encode(in_ptr, in_len, out_ptr, out_len);
}
