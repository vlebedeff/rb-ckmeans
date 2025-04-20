#include <stdio.h>
#include <assert.h>
#include "ruby.h"

#define ARENA_MIN_CAPACITY 1024

VALUE rb_return_nil(VALUE self);
VALUE rb_xsorted_cluster_index(VALUE self);

typedef struct Arena {
    int64_t capacity;
    int64_t offset;
    uint8_t *buffer;
} Arena;

typedef struct MatrixF {
    int64_t ncols;
    int64_t nrows;
    long double *values;
} MatrixF;

typedef struct MatrixI {
    int64_t ncols;
    int64_t nrows;
    int64_t *values;
} MatrixI;

typedef struct VectorF {
    int64_t nvalues;
    long double *values;
} VectorF;

typedef struct VectorI {
    int64_t nvalues;
    int64_t *values;
} VectorI;

typedef struct State {
    int64_t xcount;
    Arena   *arena;
    MatrixF *cost;
    MatrixI *splits;
    VectorF *xsum;
    VectorF *xsumsq;
} State;

/* TODO: validate these are all non negative */
typedef struct RowParams {
    int64_t row;
    int64_t imin;
    int64_t imax;
    int64_t istep;
} RowParams;

typedef struct {
    long double mean;
    long double variance;
} SegmentStats;

Arena       *arena_create(uint64_t);
void        *arena_alloc(Arena*, int64_t);
void         arena_rewind(Arena*);
void         arena_destroy(Arena*);

MatrixF     *matrix_create_f(Arena*, int64_t, int64_t);
MatrixI     *matrix_create_i(Arena*, int64_t, int64_t);
void         matrix_set_f(MatrixF*, int64_t, int64_t, long double value);
long double  matrix_get_f(MatrixF*, int64_t, int64_t);
void         matrix_inspect_f(MatrixF*);
void         matrix_set_i(MatrixI*, int64_t, int64_t, int64_t value);
int64_t      matrix_get_i(MatrixI*, int64_t, int64_t);
void         matrix_inspect_i(MatrixI*);

VectorF     *vector_create_f(Arena*, int64_t);
void         vector_set_f(VectorF*, int64_t offset, long double value);
long double  vector_get_f(VectorF*, int64_t offset);
long double  vector_get_diff_f(VectorF*, int64_t, int64_t);
void         vector_inspect_f(VectorF*);
VectorI     *vector_create_i(Arena*, int64_t);
VectorI     *vector_dup_i(VectorI*, Arena*);
void         vector_set_i(VectorI*, int64_t offset, int64_t value);
int64_t      vector_get_i(VectorI*, int64_t offset);
int64_t      vector_get_diff_i(VectorI*, int64_t, int64_t);
void         vector_downsize_i(VectorI*, int64_t);
void         vector_inspect_i(VectorI*);

long double  dissimilarity(int64_t, int64_t, VectorF*, VectorF*);
void         fill_row(State, int64_t, int64_t, int64_t);
void         smawk(State, RowParams, VectorI*);
void         find_min_from_candidates(State, RowParams, VectorI*);
VectorI     *prune_candidates(State, RowParams, VectorI*);
void         fill_even_positions(State, RowParams, VectorI*);
SegmentStats shifted_data_variance(VectorF*, int64_t, int64_t);
VectorI     *backtrack_sizes(State, int64_t);

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
    State    state   = {
        .arena  = arena,
        .xcount = xcount,
        .cost   = cost,
        .splits = splits,
        .xsum   = xsum,
        .xsumsq = xsumsq
    };

    for (int64_t i = 0; i < xcount; i++) {
        long double xi = NUM2DBL(rb_ary_entry(rb_xsorted, i));
        vector_set_f(xsorted, i, xi);
    }

    /* printf("XSORTED \t"); vector_inspect_f(xsorted); */

    long double shift        = vector_get_f(xsorted, xcount / 2);
    long double diff_initial = vector_get_f(xsorted, 0) - shift;

    vector_set_f(xsum, 0, diff_initial);
    vector_set_f(xsumsq, 0, diff_initial * diff_initial);

    for (int64_t i = 1; i < xcount; i++) {
        long double xi          = vector_get_f(xsorted, i);
        long double xsum_prev   = vector_get_f(xsum, i - 1);
        long double xsumsq_prev = vector_get_f(xsumsq, i - 1);
        long double diff        = xi - shift;

        vector_set_f(xsum, i, xsum_prev + diff);
        vector_set_f(xsumsq, i, xsumsq_prev + diff * diff);
        matrix_set_f(cost, 0, i, dissimilarity(0, i, xsum, xsumsq));
        matrix_set_i(splits, 0, i, 0);
    }

    for (int64_t q = 1; q <= kmax - 1; q++) {
        int64_t imin = (q < kmax - 1) ? ((q > 1) ? q : 1) : xcount - 1;
        fill_row(state, q, imin, xcount - 1);
    }

    printf("FINAL COST\n"); matrix_inspect_f(cost);
    printf("FINAL SPLITS\n"); matrix_inspect_i(splits);

    arena_destroy(arena);

    return Qnil;
}

VectorI *backtrack_sizes(State state, int64_t k)
{
    int64_t xcount  = state.xcount;
    MatrixI *splits = state.splits;
    VectorI *sizes  = vector_create_i(state.arena, k);

    for (int64_t i = k - 1, right = xcount - 1, left = 0; i >= 0; i--) {
        left = matrix_get_i(splits, i, right);

        vector_set_i(sizes, i, right - left + 1);

        if (i > 0) right = left - 1;
    }
    return sizes;
}

SegmentStats shifted_data_variance(VectorF *xsorted, int64_t left, int64_t right)
{
    const int64_t n    = right - left + 1;
    long double sum    = 0.0;
    long double sumsq  = 0.0;
    SegmentStats stats = { .mean = 0.0, .variance = 0.0 };

    if (right >= left) {
        const long double median = vector_get_f(xsorted, (left + right) / 2);

        for (int64_t i = left; i <= right; i++) {
            const long double sumi = vector_get_f(xsorted, i) - median;

            sum   += sumi;
            sumsq += sumi * sumi;
        }

        stats.mean = (sum / n) + median;
        if (n > 1) {
            stats.variance = (sumsq - (sum * sum / n)) / (n - 1);
        }
    }

    return stats;
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
    const int64_t imin  = rparams.imin;
    const int64_t imax  = rparams.imax;
    const int64_t istep = rparams.istep;

    if ((imax - imin) <= (0 * istep)) {
        find_min_from_candidates(state, rparams, split_candidates);
    } else {

        VectorI *odd_candidates = prune_candidates(state, rparams, split_candidates);
        /* printf("PRUNED\t"); vector_inspect_i(odd_candidates); */
        int64_t istepx2         = istep * 2;
        int64_t imin_odd        = imin + istep;
        int64_t imax_odd        = imin_odd + ((imax - imin_odd) / istepx2 * istepx2);
        RowParams rparams_odd   = { .row = rparams.row, .imin = imin_odd, .imax = imax_odd, .istep = istepx2 };

        smawk(state, rparams_odd, odd_candidates);
        fill_even_positions(state, rparams, split_candidates);
    }
}

void fill_even_positions(State state, RowParams rparams, VectorI *split_candidates)
{
    int64_t row     = rparams.row;
    int64_t imin    = rparams.imin;
    int64_t imax    = rparams.imax;
    int64_t istep   = rparams.istep;
    int64_t n       = split_candidates->nvalues;
    int64_t istepx2 = istep * 2;
    int64_t jl      = vector_get_i(split_candidates, 0);
    VectorF *xsum   = state.xsum;
    VectorF *xsumsq = state.xsumsq;
    MatrixI *splits = state.splits;

    for (int64_t i = imin, r = 0; i <= imax; i += istepx2) {
        while (vector_get_i(split_candidates, r) < jl) r++;

        int64_t rcandidate    = vector_get_i(split_candidates, r);
        int64_t cost_base_row = row - 1;
        int64_t cost_base_col = rcandidate - 1;
        long double cost      =
            matrix_get_f(state.cost, cost_base_row, cost_base_col) + dissimilarity(rcandidate, i, xsum, xsumsq);

        matrix_set_f(state.cost, row, i, cost);
        matrix_set_i(state.splits, row, i, rcandidate);

        int64_t jh         = (i + istep) <= imax ? matrix_get_i(splits, row, i + istep) : vector_get_i(split_candidates, n - 1);
        int64_t jmax       = jh < i ? jh : i;
        long double sjimin = dissimilarity(jmax, i, xsum, xsumsq);

        for (++r; r < n && vector_get_i(split_candidates, r) <= jmax; r++) {
            int64_t jabs = vector_get_i(split_candidates, r);

            if (jabs > i) break;
            if (jabs < matrix_get_i(splits, row - 1, i)) continue;

            long double cost_base = matrix_get_f(state.cost, row - 1, jabs  - 1);
            long double sj        = cost_base + dissimilarity(jabs, i, xsum, xsumsq);
            long double cost_prev = matrix_get_f(state.cost, row, i);

            if (sj <= cost_prev) {
                matrix_set_f(state.cost, row, i, sj);
                matrix_set_i(state.splits, row, i, jabs);
            } else if (cost_base + sjimin > cost_prev) {
                break;
            }
        }

        r--;
        jl = jh;
    }
}

void find_min_from_candidates(State state, RowParams rparams, VectorI *split_candidates)
{
    const int64_t row     = rparams.row;
    const int64_t imin    = rparams.imin;
    const int64_t imax    = rparams.imax;
    const int64_t istep   = rparams.istep;
    MatrixF *const cost   = state.cost;
    MatrixI *const splits = state.splits;

    int64_t optimal_split_idx_prev = 0;

    for (int64_t i = imin; i <= imax; i += istep)
    {
        const int64_t optimal_split_idx = optimal_split_idx_prev;
        const int64_t optimal_split     = vector_get_i(split_candidates, optimal_split_idx);
        const int64_t cost_prev         = matrix_get_f(cost, row - 1, optimal_split - 1);
        const long double added_cost    = dissimilarity(optimal_split, i, state.xsum, state.xsumsq);

        matrix_set_f(cost, row, i, cost_prev + added_cost);
        matrix_set_i(splits, row, i, optimal_split);

        for (int64_t r = optimal_split_idx + 1; r < split_candidates->nvalues; r++)
        {
            int64_t split = vector_get_i(split_candidates, r);

            if (split < matrix_get_i(splits, row - 1, i)) continue;
            if (split > i) break;

            long double split_cost =
                matrix_get_f(cost, row - 1, split - 1) + dissimilarity(split, i, state.xsum, state.xsumsq);

            if (split_cost > matrix_get_f(cost, row, i)) continue;

            matrix_set_f(cost, row, i, split_cost);
            matrix_set_i(splits, row, i, split);
            optimal_split_idx_prev = r;
        }
    }
}

VectorI *prune_candidates(State state, RowParams rparams, VectorI *split_candidates)
{
    int64_t n = ((rparams.imax - rparams.imin) / rparams.istep) + 1;
    int64_t m = split_candidates->nvalues;

    if (n >= m) return split_candidates;

    int64_t left    = -1;
    int64_t right   = 0;
    VectorI *pruned = vector_dup_i(split_candidates, state.arena);

    while (m > n)
    {
        int64_t p         = left + 1;
        int64_t i         = rparams.imin + p * rparams.istep;
        int64_t j         = vector_get_i(pruned, right);
        int64_t jnext     = vector_get_i(pruned, right + 1);
        long double sl    =
            matrix_get_f(state.cost, rparams.row - 1, j - 1) + dissimilarity(j, i, state.xsum, state.xsumsq);
        long double snext =
            matrix_get_f(state.cost, rparams.row - 1, jnext - 1) + dissimilarity(jnext, i, state.xsum, state.xsumsq);

        if ((sl < snext) && (p < n - 1)) {
            left++;
            right++;
            vector_set_i(pruned, left, j);
        } else if ((sl < snext) && (p == n - 1)) {
            right++;
            m--;
            vector_set_i(pruned, right, j);
        } else {
            if (p > 0) {
                /* TODO: extract `vector_setcpy_T` */
                vector_set_i(pruned, right, vector_get_i(pruned, left));
                left--;
            } else {
                right++;
            }

            m--;
        }
    }

    for (int64_t i = left + 1; i < m; i++) {
        /* TODO: extract `vector_setcpy_T` */
        vector_set_i(pruned, i, vector_get_i(pruned, right++));
    }

    vector_downsize_i(pruned, m);

    return pruned;
}

long double dissimilarity(int64_t j, int64_t i, VectorF *xsum, VectorF *xsumsq) {
    long double sji = 0.0;

    if (j >= i) return sji;

    if (j > 0) {
        /* TODO: looks more like `segment_delta` */
        long double segment_sum = vector_get_diff_f(xsum, i, j - 1);
        int64_t segment_size    = i - j + 1;
        sji                     = vector_get_diff_f(xsumsq, i, j - 1) - (segment_sum * segment_sum / segment_size);
    } else {
        long double xsumi = vector_get_f(xsum, i);
        sji               = vector_get_f(xsumsq, i) - (xsumi * xsumi / (i + 1));
    }

    return (sji > 0) ? sji : 0.0;
}

VectorF *vector_create_f(Arena *arena, int64_t nvalues) {
    VectorF *v;

    v          = arena_alloc(arena, sizeof(*v));
    v->values  = arena_alloc(arena, sizeof(*(v->values)) * nvalues);
    v->nvalues = nvalues;

    return v;
}

VectorI *vector_create_i(Arena *arena, int64_t nvalues) {
    VectorI *v;

    v          = arena_alloc(arena, sizeof(*v));
    v->values  = arena_alloc(arena, sizeof(*(v->values)) * nvalues);
    v->nvalues = nvalues;

    return v;
}

VectorI *vector_dup_i(VectorI *v, Arena *arena)
{
    VectorI *vdup = vector_create_i(arena, v->nvalues);

    /* TODO: use one memcpy call */
    for (int64_t i = 0; i < v->nvalues; i++) {
        vector_set_i(vdup, i, vector_get_i(v, i));
    }

    return vdup;
}

void vector_set_f(VectorF *v, int64_t offset, long double value) {
    assert(offset < v->nvalues && "[vector_set_f] element index should be less than nvalues");

    *(v->values + offset) = value;
}

void vector_set_i(VectorI *v, int64_t offset, int64_t value) {
    assert(offset < v->nvalues && "[vector_set_i] element index should be less than nvalues");

    *(v->values + offset) = value;
}

int64_t vector_get_i(VectorI *v, int64_t offset) {
    assert(offset < v->nvalues && "[vector_get_i] element index should be less than nvalues");

    return *(v->values + offset);
}

int64_t vector_get_diff_i(VectorI *v, int64_t i, int64_t j) {
    assert(i < v->nvalues && "[vector_get_diff_i] i should be less than nvalues");
    assert(j < v->nvalues && "[vector_get_diff_i] j should be less than nvalues");

    return *(v->values + i) - *(v->values + j);
}

void vector_downsize_i(VectorI *v, int64_t new_size) {
    v->nvalues = new_size;
}

void vector_inspect_i(VectorI *v) {
    for (int64_t i = 0; i < v->nvalues - 1; i++) {
        printf("%lld, ", vector_get_i(v, i));
    }
    printf("%lld\n", vector_get_i(v, v->nvalues - 1));
}

long double vector_get_f(VectorF *v, int64_t offset) {
    assert(offset < v->nvalues && "[vector_get_f] element index should be less than nvalues");

    return *(v->values + offset);
}

long double vector_get_diff_f(VectorF *v, int64_t i, int64_t j) {
    assert(i < v->nvalues && "[vector_get_diff_f] i should be less than nvalues");
    assert(j < v->nvalues && "[vector_get_diff_f] j should be less than nvalues");

    return *(v->values + i) - *(v->values + j);
}

void vector_inspect_f(VectorF *v) {
    for (int64_t i = 0; i < v->nvalues - 1; i++) {
        printf("%Lf, ", vector_get_f(v, i));
    }
    printf("%Lf\n", vector_get_f(v, v->nvalues - 1));
}

MatrixF *matrix_create_f(Arena *arena, int64_t nrows, int64_t ncols) {
    MatrixF *m;

    /* TODO: use one allocation */
    m         = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * ncols * nrows);
    m->ncols  = ncols;
    m->nrows  = nrows;

    return m;
}

MatrixI *matrix_create_i(Arena *arena, int64_t nrows, int64_t ncols) {
    MatrixI *m;

    /* TODO: use one allocation */
    m         = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * ncols * nrows);
    m->ncols  = ncols;
    m->nrows  = nrows;

    return m;
}

void matrix_set_f(MatrixF *m, int64_t i, int64_t j, long double value) {
    assert(i < m->nrows && "[matrix_set_f] row offset should be less than nrows");
    assert(j < m->cols &&  "[matrix_set_f] col offset should be less than ncols");

    int64_t offset = i * m->ncols + j;
    *(m->values + offset) = value;
}

long double matrix_get_f(MatrixF *m, int64_t i, int64_t j) {
    assert(i < m->nrows && "[matrix_get_f] row offset should be less than nrows");
    assert(j < m->cols &&  "[matrix_get_f] col offset should be less than ncols");

    int64_t offset = i * m->ncols + j;
    return *(m->values + offset);
}

void matrix_inspect_f(MatrixF *m) {
    for (int64_t i = 0; i < m->nrows; i++) {
        for (int64_t j = 0; j < m->ncols - 1; j++) {
            long double value = matrix_get_f(m, i, j);

            printf("%Lf, ", value);
        }
        printf("%Lf\n", matrix_get_f(m, i, m->ncols - 1));
    }
}

void matrix_inspect_i(MatrixI *m) {
    for (int64_t i = 0; i < m->nrows; i++) {
        for (int64_t j = 0; j < m->ncols - 1; j++)
            printf("%lld, ", matrix_get_i(m, i, j));
        printf("%lld\n", matrix_get_i(m, i, m->ncols - 1));
    }
}

void matrix_set_i(MatrixI *m, int64_t i, int64_t j, int64_t value) {
    assert(i < m->nrows && "[matrix_set_i] row offset should be less than nrows");
    assert(j < m->cols &&  "[matrix_set_i] col offset should be less than ncols");

    int64_t offset = i * m->ncols + j;
    *(m->values + offset) = value;
}

int64_t matrix_get_i(MatrixI *m, int64_t i, int64_t j) {
    assert(i < m->nrows && "[matrix_get_i] row offset should be less than nrows");
    assert(j < m->cols &&  "[matrix_get_i] col offset should be less than ncols");

    int64_t offset = i * m->ncols + j;
    return *(m->values + offset);
}

Arena *arena_create(uint64_t capacity) {
    if (capacity < ARENA_MIN_CAPACITY) {
        capacity = ARENA_MIN_CAPACITY;
    }

    Arena *arena;

    arena        = malloc(sizeof(*arena));
    void *buffer = calloc(1, capacity);

    if (!arena || !buffer) {
        printf("Failed to allocate arena\n");

        if (arena) free(arena);
        if (buffer) free(buffer);

        return NULL;
    }

    arena->buffer   = buffer;
    arena->capacity = capacity;
    arena->offset   = 0;

    printf("[Arena Created] Capacity: %lld, offset: %lld\n", arena->capacity, arena->offset);

    return arena;
}

void *arena_alloc(Arena *arena, int64_t size) {
    size = (size + 7) & ~7;

    if (arena->offset + size > arena->capacity) {
        return NULL;
    }

    void *ptr = arena->buffer + arena->offset;
    arena->offset += size;

    return ptr;
}

void arena_destroy(Arena *arena) {
    printf("[Arena Destroy] Capacity: %lld, offset: %lld, left: %lld\n", arena->capacity, arena->offset, arena->capacity - arena->offset);
    free(arena->buffer);
    free(arena);
}
