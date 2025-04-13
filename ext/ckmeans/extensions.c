#include <stdio.h>
#include "ruby.h"

#define ARENA_MIN_CAPACITY 1024

VALUE rb_return_nil(VALUE self);
VALUE rb_xsorted_cluster_index(VALUE self);

typedef struct Arena {
    size_t   capacity;
    size_t   offset;
    uint8_t *buffer;
} Arena;

typedef struct MatrixF {
    uint64_t ncols;
    uint64_t nrows;
    long double *values;
} MatrixF;

typedef struct MatrixI {
    uint64_t ncols;
    uint64_t nrows;
    int64_t *values;
} MatrixI;

typedef struct VectorF {
    uint64_t nvalues;
    long double *values;
} VectorF;

typedef struct VectorI {
    uint64_t nvalues;
    int64_t *values;
} VectorI;

typedef struct State {
    Arena   *arena;
    MatrixF *cost;
    MatrixI *breaks;
    VectorF *xsum;
    VectorF *xsumsq;
} State;

Arena *arena_create(uint64_t);
void  *arena_alloc(Arena*, size_t);
void   arena_rewind(Arena*);
void   arena_destroy(Arena*);

MatrixF     *matrix_create_f(Arena*, uint64_t, uint64_t);
MatrixI     *matrix_create_i(Arena*, uint64_t, uint64_t);
void         matrix_set_f(MatrixF*, uint64_t, uint64_t, long double value);
void         matrix_set_i(MatrixI*, uint64_t, uint64_t, int64_t value);

VectorF     *vector_create_f(Arena*, uint64_t);
VectorI     *vector_create_i(Arena*, uint64_t);
void         vector_set_f(VectorF*, uint64_t offset, long double value);
void         vector_set_i(VectorI*, uint64_t offset, int64_t value);
long double  vector_get_f(VectorF*, uint64_t offset);

long double  dissimilarity(uint64_t, uint64_t, VectorF*, VectorF*);
void         fill_row(State, uint64_t, uint64_t, uint64_t);
void         smawk(State, uint64_t, uint64_t, uint64_t, uint64_t, VectorI*);

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
    VALUE rb_xcount  = rb_ivar_get(self, rb_intern("@xcount"));
    /* VALUE rb_kmin    = rb_ivar_get(self, rb_intern("@kmin")); */
    VALUE rb_kmax    = rb_ivar_get(self, rb_intern("@kmax"));
    VALUE rb_xsorted = rb_ivar_get(self, rb_intern("@xsorted"));
    uint64_t xcount  = NUM2ULL(rb_xcount);
    /* uint64_t kmin    = NUM2ULL(rb_kmin); */
    uint64_t kmax    = NUM2ULL(rb_kmax);
    Arena *arena     = arena_create(xcount * kmax * ALLOCATION_FACTOR);

    if (arena == NULL) {
        return Qnil;
    }

    MatrixF *cost    = matrix_create_f(arena, kmax, xcount);
    MatrixI *breaks  = matrix_create_i(arena, kmax, xcount);
    VectorF *xsorted = vector_create_f(arena, xcount);
    VectorF *xsum    = vector_create_f(arena, xcount);
    VectorF *xsumsq  = vector_create_f(arena, xcount);
    State    state   = { .arena = arena, .cost = cost, .breaks = breaks, .xsum = xsum, .xsumsq = xsumsq };

    for (uint64_t i = 0; i < xcount; i++) {
        long double xi = NUM2DBL(rb_ary_entry(rb_xsorted, i));
        vector_set_f(xsorted, i, xi);
        printf("XSORTED[%llu]: %Lf\n", i, vector_get_f(xsorted, i));
    }

    uint64_t shift = vector_get_f(xsorted, xcount / 2);
    long double diff_initial = vector_get_f(xsorted, 0) - shift;
    vector_set_f(xsum, 0, diff_initial);
    vector_set_f(xsumsq, 0, diff_initial * diff_initial);

    for (uint64_t i = 1; i < xcount; i++) {
        long double xi = vector_get_f(xsorted, i);
        long double xsum_prev = vector_get_f(xsum, i - 1);
        long double xsumsq_prev = vector_get_f(xsumsq, i - 1);
        long double diff = xi - shift;

        vector_set_f(xsum, i, xsum_prev + diff);
        vector_set_f(xsumsq, i, xsumsq_prev + diff * diff);
        matrix_set_f(cost, 0, i, dissimilarity(0, i, xsum, xsumsq));
        matrix_set_i(breaks, 0, i, 0);
    }


    for (uint64_t q = 1; q < kmax - 1; q++) {
        uint64_t imin = (q < kmax - 1) ? ((q > 1) ? q : 1) : xcount - 1;
        fill_row(state, q, imin, xcount - 1);
    }

    arena_destroy(arena);

    return Qnil;
}

void fill_row(State state, uint64_t q, uint64_t imin, uint64_t imax) {
    uint64_t size = imax - q + 1;
    VectorI *split_candidates = vector_create_i(state.arena, size);
    for (uint64_t i = 0; i < size; i++) {
        vector_set_i(split_candidates, i, q + i);
    }
    /* smawk(state, imin, imax, 1, q, split_candidates); */
}

long double dissimilarity(uint64_t i, uint64_t j, VectorF *xsum, VectorF *xsumsq) {
    long double sji = 0.0;

    if (j >= i) return sji;

    if (j > 0) {
        long double segment_sum = vector_get_f(xsum, i) - vector_get_f(xsum, j - 1);
        int64_t segment_size = i - j + 1;
        sji = vector_get_f(xsumsq, i) - vector_get_f(xsumsq, j - 1) - (segment_sum * segment_sum / segment_size);
    } else {
        long double xsumi = vector_get_f(xsum, i);
        sji = vector_get_f(xsumsq, i) - (xsumi * xsumi / (i + 1));
    }

    return (sji > 0) ? sji : 0.0;
}

VectorF *vector_create_f(Arena *arena, uint64_t nvalues) {
    VectorF *v;

    v = arena_alloc(arena, sizeof(*v));
    v->values = arena_alloc(arena, sizeof(*(v->values)) * nvalues);
    v->nvalues = nvalues;

    return v;
}

VectorI *vector_create_i(Arena *arena, uint64_t nvalues) {
    VectorI *v;

    v = arena_alloc(arena, sizeof(*v));
    v->values = arena_alloc(arena, sizeof(*(v->values)) * nvalues);
    v->nvalues = nvalues;

    return v;
}

void vector_set_f(VectorF *v, uint64_t offset, long double value) {
    if (offset < 0 || offset >= v->nvalues) {
        printf("[Vector] %llu is out bounds", offset);
        return;
    }

    *(v->values + offset) = value;
}

void vector_set_i(VectorI *v, uint64_t offset, int64_t value) {
    if (offset < 0 || offset >= v->nvalues) {
        printf("[Vector] %llu is out bounds", offset);
        return;
    }

    *(v->values + offset) = value;
}

long double vector_get_f(VectorF *v, uint64_t offset) {
    if (offset < 0 || offset >= v->nvalues) {
        printf("[Vector] %llu is out bounds", offset);
        return 0;
    }

    return *(v->values + offset);
}

MatrixF *matrix_create_f(Arena *arena, uint64_t ncols, uint64_t nrows) {
    MatrixF *m;

    m = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * ncols * nrows);
    m->ncols = ncols;
    m->nrows = nrows;

    return m;
}

MatrixI *matrix_create_i(Arena *arena, uint64_t ncols, uint64_t nrows) {
    MatrixI *m;

    m = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * ncols * nrows);
    m->ncols = ncols;
    m->nrows = nrows;

    return m;
}

void matrix_set_f(MatrixF *m, uint64_t i, uint64_t j, long double value) {
    if (i < 0 || i >= m->nrows) {
        printf("[matrix_set_f] i=%llu is out of bounds\n", i);
        return;
    }

    if (j < 0 || i >= m->ncols) {
        printf("[matrix_set_f] j=%llu is out of bounds\n", j);
        return;
    }

    uint64_t offset = i * m->ncols + j;
    *(m->values + offset) = value;
}

void matrix_set_i(MatrixI *m, uint64_t i, uint64_t j, int64_t value) {
    if (i < 0 || i >= m->nrows) {
        printf("[matrix_set_f] i=%llu is out of bounds\n", i);
        return;
    }

    if (j < 0 || i >= m->ncols) {
        printf("[matrix_set_f] j=%llu is out of bounds\n", j);
        return;
    }

    uint64_t offset = i * m->ncols + j;
    *(m->values + offset) = value;
}

Arena *arena_create(uint64_t capacity) {
    if (capacity < ARENA_MIN_CAPACITY) {
        capacity = ARENA_MIN_CAPACITY;
    }

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
