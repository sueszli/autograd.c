#include "json.h"
#include "mem.h"
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>

static alignas(16) u8 tree_buf[4 * 1024 * 1024]; // parity.json parses into about 3.8 MB
static Arena tree = {.buf = tree_buf, .cap = sizeof(tree_buf)};

static void *tree_alloc(usize bytes) {
    // cJSON's malloc hook
    return arena_alloc(&tree, bytes);
}

static void tree_free(void *p) {
    // cJSON frees without a length, so the tree lives until exit
    (void)p;
}

const cJSON *json_load(const char *path) {
    // read into a static buffer with plain syscalls, stdio would malloc its own buffer
    static char raw[512 * 1024];
    int fd = open(path, O_RDONLY);
    assert(fd >= 0 && "cannot open json file");
    ssize_t size = read(fd, raw, sizeof(raw) - 1);
    assert(size > 0 && size < (ssize_t)sizeof(raw) - 1 && "cannot read json file, or it outgrew its buffer");
    int close_ok = close(fd);
    assert(close_ok == 0 && "cannot close json file");
    cJSON_InitHooks(&(cJSON_Hooks){.malloc_fn = tree_alloc, .free_fn = tree_free});
    const cJSON *j = cJSON_Parse(raw);
    assert(j != NULL && "cannot parse json file");
    return j;
}

const cJSON *json_get(const cJSON *j, const char *key) {
    // case-sensitive lookup
    const cJSON *f = cJSON_GetObjectItemCaseSensitive(j, key);
    assert(f != NULL && "json key missing");
    return f;
}

void json_matrix(const cJSON *j, f64 *out, i64 rows, i64 cols, bool transposed) {
    // walk the linked lists in order, writing each value to its spot in out
    i64 outer = transposed ? cols : rows, inner = transposed ? rows : cols;
    assert(cJSON_GetArraySize(j) == outer && "json matrix has the wrong number of rows");
    const cJSON *row = j->child;
    for (i64 r = 0; r < outer; r++, row = row->next) {
        assert(cJSON_GetArraySize(row) == inner && "json matrix has the wrong number of columns");
        const cJSON *v = row->child;
        for (i64 c = 0; c < inner; c++, v = v->next) {
            out[transposed ? c * cols + r : r * cols + c] = v->valuedouble;
        }
    }
}
