#include <stdio.h>
#include <assert.h>
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
    size_t ncols;
    size_t nrows;
    long double *values;
} MatrixF;

typedef struct MatrixI {
    size_t ncols;
    size_t nrows;
    int64_t *values;
} MatrixI;

typedef struct VectorF {
    size_t nvalues;
    long double *values;
} VectorF;

typedef struct VectorI {
    size_t nvalues;
    int64_t *values;
} VectorI;

typedef struct State {
    Arena   *arena;
    MatrixF *cost;
    MatrixI *splits;
    VectorF *xsum;
    VectorF *xsumsq;
} State;

typedef struct RowParams {
    int64_t row;
    int64_t imin;
    int64_t imax;
    int64_t istep;
} RowParams;

Arena       *arena_create(uint64_t);
void        *arena_alloc(Arena*, size_t);
void         arena_rewind(Arena*);
void         arena_destroy(Arena*);

MatrixF     *matrix_create_f(Arena*, size_t, size_t);
MatrixI     *matrix_create_i(Arena*, size_t, size_t);
void         matrix_set_f(MatrixF*, size_t, size_t, long double value);
long double  matrix_get_f(MatrixF*, size_t, size_t);
void         matrix_set_i(MatrixI*, size_t, size_t, int64_t value);

VectorF     *vector_create_f(Arena*, size_t);
VectorI     *vector_create_i(Arena*, size_t);
void         vector_set_f(VectorF*, size_t offset, long double value);
void         vector_set_i(VectorI*, size_t offset, int64_t value);
int64_t      vector_get_i(VectorI*, size_t offset);
int64_t      vector_get_diff_i(VectorI*, size_t, size_t);
long double  vector_get_f(VectorF*, size_t offset);
long double  vector_get_diff_f(VectorF*, size_t, size_t);

long double  dissimilarity(int64_t, int64_t, VectorF*, VectorF*);
void         fill_row(State, int64_t, int64_t, int64_t);
void         smawk(State, RowParams, VectorI*);
void         find_min_from_candidates(State, RowParams, VectorI*);

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
    int64_t xcount   = NUM2LL(rb_xcount);
    /* int64_t kmin    = NUM2LL(rb_kmin); */
    int64_t kmax     = NUM2LL(rb_kmax);
    Arena *arena     = arena_create(xcount * kmax * ALLOCATION_FACTOR);

    if (arena == NULL) {
        return Qnil;
    }

    MatrixF *cost    = matrix_create_f(arena, kmax, xcount);
    MatrixI *splits  = matrix_create_i(arena, kmax, xcount);
    VectorF *xsorted = vector_create_f(arena, xcount);
    VectorF *xsum    = vector_create_f(arena, xcount);
    VectorF *xsumsq  = vector_create_f(arena, xcount);
    State    state   = { .arena = arena, .cost = cost, .splits = splits, .xsum = xsum, .xsumsq = xsumsq };

    for (int64_t i = 0; i < xcount; i++) {
        long double xi = NUM2DBL(rb_ary_entry(rb_xsorted, i));
        vector_set_f(xsorted, i, xi);
        printf("XSORTED[%llu]: %Lf\n", i, vector_get_f(xsorted, i));
    }

    int64_t shift = vector_get_f(xsorted, xcount / 2);
    long double diff_initial = vector_get_f(xsorted, 0) - shift;
    vector_set_f(xsum, 0, diff_initial);
    vector_set_f(xsumsq, 0, diff_initial * diff_initial);

    for (int64_t i = 1; i < xcount; i++) {
        long double xi = vector_get_f(xsorted, i);
        long double xsum_prev = vector_get_f(xsum, i - 1);
        long double xsumsq_prev = vector_get_f(xsumsq, i - 1);
        long double diff = xi - shift;

        vector_set_f(xsum, i, xsum_prev + diff);
        vector_set_f(xsumsq, i, xsumsq_prev + diff * diff);
        matrix_set_f(cost, 0, i, dissimilarity(0, i, xsum, xsumsq));
        matrix_set_i(splits, 0, i, 0);
    }


    for (int64_t q = 1; q < kmax - 1; q++) {
        int64_t imin = (q < kmax - 1) ? ((q > 1) ? q : 1) : xcount - 1;
        fill_row(state, q, imin, xcount - 1);
    }

    arena_destroy(arena);

    return Qnil;
}

void fill_row(State state, int64_t q, int64_t imin, int64_t imax) {
    int64_t size = imax - q + 1;
    VectorI *split_candidates = vector_create_i(state.arena, size);
    for (int64_t i = 0; i < size; i++) {
        vector_set_i(split_candidates, i, q + i);
    }
    RowParams rparams = { .row = q, .imin = imin, .imax = imax, .istep = 1 };
    smawk(state, rparams, split_candidates);
}

void smawk(State state, RowParams rparams, VectorI *split_candidates) {
    if ((rparams.imax - rparams.imin) <= (0 * rparams.istep)) {
        find_min_from_candidates(state, rparams, split_candidates);
    } else {
        /* NOT IMPLEMENTED */
        return;
    }
}

void find_min_from_candidates(State state, RowParams rparams, VectorI *split_candidates) {
    int64_t rmin_prev = 0;

    for (int64_t i = rparams.imin; i <= rparams.imax; i += rparams.istep) {
        int64_t rmin = rmin_prev;
        int64_t split_candidate = vector_get_i(split_candidates, rmin);
        int64_t cost_prev = matrix_get_f(state.cost, rparams.row - 1, split_candidate - 1);
        long double added_cost = dissimilarity(split_candidate, i, state.xsum, state.xsumsq);

        matrix_set_f(state.cost, rparams.row, i, cost_prev + added_cost);
        matrix_set_i(state.splits, rparams.row, i, split_candidate);

        for (size_t r = rmin + 1; r < split_candidates->nvalues; r++) {
            int64_t split = vector_get_i(split_candidates, r);

            if (split < matrix_get_f(state.cost, rparams.row - 1, i)) continue;
            if (split > i) break;

            long double split_cost =
                matrix_get_f(state.cost, rparams.row - 1, split - 1) + dissimilarity(split, i, state.xsum, state.xsumsq);

            if (split_cost <= matrix_get_f(state.cost, rparams.row, i)) continue;

            matrix_set_f(state.cost, rparams.row, i, split_cost);
            matrix_set_i(state.splits, rparams.row, i, split);
            rmin_prev = r;
        }
    }
}

long double dissimilarity(int64_t i, int64_t j, VectorF *xsum, VectorF *xsumsq) {
    long double sji = 0.0;

    if (j >= i) return sji;

    if (j > 0) {
        long double segment_sum = vector_get_diff_f(xsum, i, j - 1);
        int64_t segment_size = i - j + 1;
        sji = vector_get_diff_f(xsumsq, i, j - 1) - (segment_sum * segment_sum / segment_size);
    } else {
        long double xsumi = vector_get_f(xsum, i);
        sji = vector_get_f(xsumsq, i) - (xsumi * xsumi / (i + 1));
    }

    return (sji > 0) ? sji : 0.0;
}

VectorF *vector_create_f(Arena *arena, size_t nvalues) {
    VectorF *v;

    v = arena_alloc(arena, sizeof(*v));
    v->values = arena_alloc(arena, sizeof(*(v->values)) * nvalues);
    v->nvalues = nvalues;

    return v;
}

VectorI *vector_create_i(Arena *arena, size_t nvalues) {
    VectorI *v;

    v = arena_alloc(arena, sizeof(*v));
    v->values = arena_alloc(arena, sizeof(*(v->values)) * nvalues);
    v->nvalues = nvalues;

    return v;
}

void vector_set_f(VectorF *v, size_t offset, long double value) {
    assert(offset < v->nvalues && "[vector_set_f] element index should be less than nvalues");

    *(v->values + offset) = value;
}

void vector_set_i(VectorI *v, size_t offset, int64_t value) {
    assert(offset < v->nvalues && "[vector_set_i] element index should be less than nvalues");

    *(v->values + offset) = value;
}

int64_t vector_get_i(VectorI *v, size_t offset) {
    assert(offset < v->nvalues && "[vector_get_i] element index should be less than nvalues");

    return *(v->values + offset);
}

int64_t vector_get_diff_i(VectorI *v, size_t i, size_t j) {
    assert(i < v->nvalues && "[vector_get_diff_i] i should be less than nvalues");
    assert(j < v->nvalues && "[vector_get_diff_i] j should be less than nvalues");

    return *(v->values + i) - *(v->values + j);
}

long double vector_get_f(VectorF *v, size_t offset) {
    assert(offset < v->nvalues && "[vector_get_f] element index should be less than nvalues");

    return *(v->values + offset);
}

long double vector_get_diff_f(VectorF *v, size_t i, size_t j) {
    assert(i < v->nvalues && "[vector_get_diff_f] i should be less than nvalues");
    assert(j < v->nvalues && "[vector_get_diff_f] j should be less than nvalues");

    return *(v->values + i) - *(v->values + j);
}

MatrixF *matrix_create_f(Arena *arena, size_t ncols, size_t nrows) {
    MatrixF *m;

    m = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * ncols * nrows);
    m->ncols = ncols;
    m->nrows = nrows;

    return m;
}

MatrixI *matrix_create_i(Arena *arena, size_t ncols, size_t nrows) {
    MatrixI *m;

    m = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * ncols * nrows);
    m->ncols = ncols;
    m->nrows = nrows;

    return m;
}

void matrix_set_f(MatrixF *m, size_t i, size_t j, long double value) {
    assert(i < m->nrows && "[matrix_set_f] row offset should be less than nrows");
    assert(j < m->cols &&  "[matrix_set_f] col offset should be less than ncols");

    size_t offset = i * m->ncols + j;
    *(m->values + offset) = value;
}

long double matrix_get_f(MatrixF *m, size_t i, size_t j) {
    assert(i < m->nrows && "[matrix_get_f] row offset should be less than nrows");
    assert(j < m->cols &&  "[matrix_get_f] col offset should be less than ncols");

    size_t offset = i * m->ncols + j;
    return *(m->values + offset);
}

void matrix_set_i(MatrixI *m, size_t i, size_t j, int64_t value) {
    assert(i < m->nrows && "[matrix_set_i] row offset should be less than nrows");
    assert(j < m->cols &&  "[matrix_set_i] col offset should be less than ncols");

    size_t offset = i * m->ncols + j;
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
