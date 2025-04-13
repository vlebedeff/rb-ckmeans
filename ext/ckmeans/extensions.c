#include <stdio.h>
#include "ruby.h"

VALUE rb_return_nil(VALUE self);
VALUE rb_xsorted_cluster_index(VALUE self);

typedef struct Arena {
    uint8_t *buffer;
    size_t   capacity;
    size_t   offset;
} Arena;

typedef struct MatrixF {
    uint64_t nrows;
    uint64_t ncols;
    long double *values;
} MatrixF;

typedef struct MatrixI {
    uint64_t nrows;
    uint64_t ncols;
    int64_t *values;
} MatrixI;

Arena *arena_create(uint64_t);
void  *arena_alloc(Arena*, size_t);
void   arena_rewind(Arena*);
void   arena_destroy(Arena*);

MatrixF *matrixf_create(Arena*, uint64_t, uint64_t);
MatrixI *matrixi_create(Arena*, uint64_t, uint64_t);

void Init_extensions(void) {
    VALUE ckmeans_module = rb_const_get(rb_cObject, rb_intern("Ckmeans"));
    VALUE clusterer_class = rb_const_get(ckmeans_module, rb_intern("Clusterer"));

    rb_define_singleton_method(ckmeans_module, "c_do_nothing", rb_return_nil, 0);
    rb_define_method(clusterer_class, "xsorted_cluster_index", rb_xsorted_cluster_index, 0);
}

VALUE rb_return_nil(VALUE self) {
    return Qnil;
}

# define ALLOCATION_FACTOR 100

VALUE rb_xsorted_cluster_index(VALUE self) {
    VALUE rb_xcount = rb_ivar_get(self, rb_intern("@xcount"));
    VALUE rb_kmin = rb_ivar_get(self, rb_intern("@kmin"));
    VALUE rb_kmax = rb_ivar_get(self, rb_intern("@kmax"));

    uint64_t xcount = NUM2ULL(rb_xcount);
    uint64_t kmin = NUM2ULL(rb_kmin);
    uint64_t kmax = NUM2ULL(rb_kmax);

    printf("xcount: %llu, kmin: %llu, kmax: %llu\n", xcount, kmin, kmax);

    Arena *arena = arena_create((xcount + 1) * (kmax + 1) * ALLOCATION_FACTOR);

    if (arena == NULL) {
        return Qnil;
    }

    MatrixF *smat = matrixf_create(arena, kmax, xcount);
    MatrixI *jmat = matrixi_create(arena, kmax, xcount);

    smat; // use the var to silence warnings
    jmat; // use the var to silence warnings

    arena_destroy(arena);

    return Qnil;
}

MatrixF *matrixf_create(Arena *arena, uint64_t nrows, uint64_t ncols) {
    MatrixF *m;

    m = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * nrows * ncols);

    return m;
}

MatrixI *matrixi_create(Arena *arena, uint64_t nrows, uint64_t ncols) {
    MatrixI *m;

    m = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * nrows * ncols);

    return m;
}

Arena *arena_create(uint64_t capacity) {
    Arena *arena;

    arena = malloc(sizeof(*arena));
    if (!arena) {
        printf("Failed to allocate arena\n");
        return NULL;
    }

    arena->buffer = calloc(1, capacity);
    if (!arena->buffer) {
        printf("Failed to allocate buffer\n");
        free(arena);
        return NULL;
    }

    arena->capacity = capacity;
    arena->offset = 0;
    printf("[Arena Created] Capacity: %zu, offset: %zu\n", arena->capacity, arena->offset);

    return arena;
}

void *arena_alloc(Arena *arena, size_t size) {
    size = (size + 7) & ~7;

    if (arena->offset + size > arena->capacity) {
        return NULL;
    }

    void *ptr = arena->buffer + arena->offset;
    arena->offset += size;

    return ptr;
}

void arena_destroy(Arena *arena) {
    printf("[Arena Destroy] Capacity: %zu, offset: %zu\n", arena->capacity, arena->offset);
    free(arena->buffer);
    free(arena);
}
