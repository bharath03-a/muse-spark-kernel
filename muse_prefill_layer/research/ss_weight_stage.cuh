// Research-only layout helper, not connected to production dispatch.
// RS m64k16 fragments -> SW128 BF16 m64k64 A stage, bits unchanged.
// wa must be 1024-byte aligned; column offset ki16 is one of 0,16,32,48.
__device__ __forceinline__ void ss_store_rs_fragment(
    uint32_t wa, int group, int ki16, const uint32_t (&a)[4]) {
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x & 127) >> 5;
  const int r = group * 64 + warp * 16 + (lane >> 2);
  const int c = ki16 + 2 * (lane & 3);
  const int rows[4] = {r, r + 8, r, r + 8};
  const int cols[4] = {c, c, c + 8, c + 8};
#pragma unroll
  for (int i=0; i<4; ++i) {
    uint32_t address = wa + rows[i] * 128 + cols[i] * 2;
    address ^= ((address >> 7) & 7) << 4;
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(address), "r"(a[i]) : "memory");
  }
}
