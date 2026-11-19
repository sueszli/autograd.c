#include "alloc.h"

static alignas(16) unsigned char perm_buf[512 * 1024];  // parity uses about 200 KB: two param sets, their grads and AdamW state
static alignas(16) unsigned char graph_buf[256 * 1024]; // one microGPT loss plus backward uses about 225 KB
FixedBufferAllocator perm_mem = {.buf = perm_buf, .cap = sizeof(perm_buf)};
FixedBufferAllocator graph_mem = {.buf = graph_buf, .cap = sizeof(graph_buf)};
