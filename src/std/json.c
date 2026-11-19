#include "json.h"
#include "mem.h"
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>

// the arena json_load parses into, cJSON's hooks take no context so it has to be global
static Arena *hook_arena;

static void *tree_alloc(usize bytes) {
    // cJSON's malloc hook
    return arena.alloc(hook_arena, bytes);
}

static void tree_free(void *p) {
    // cJSON frees without a length, so the tree lives until exit
    (void)p;
}

static const cJSON *json_load(const char *path, Arena *tree) {
    // read into the arena with plain syscalls, stdio would malloc its own buffer
    const usize cap = 512 * 1024;
    char *raw = arena.alloc(tree, cap);
    i32 fd = open(path, O_RDONLY);
    assert(fd >= 0 && "cannot open json file");
    ssize_t size = read(fd, raw, cap - 1);
    assert(size > 0 && size < (ssize_t)cap - 1 && "cannot read json file, or it outgrew its buffer");
    i32 close_ok = close(fd);
    assert(close_ok == 0 && "cannot close json file");
    hook_arena = tree;
    cJSON_InitHooks(&(cJSON_Hooks){.malloc_fn = tree_alloc, .free_fn = tree_free});
    const cJSON *j = cJSON_Parse(raw);
    assert(j != NULL && "cannot parse json file");
    return j;
}

static const cJSON *json_get(const cJSON *j, const char *key) {
    // case-sensitive lookup
    const cJSON *f = cJSON_GetObjectItemCaseSensitive(j, key);
    assert(f != NULL && "json key missing");
    return f;
}

static void json_matrix(const cJSON *j, f64 *out, i64 rows, i64 cols, bool transposed) {
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

const JsonNamespace json = {.load = json_load, .get = json_get, .matrix = json_matrix};
