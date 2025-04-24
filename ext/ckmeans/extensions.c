#include <stdio.h>
#include <math.h>
#include <string.h>
#include "ruby.h"

typedef struct Arena {
    uint32_t capacity;
    uint32_t offset;
    uint8_t  *buffer;
} Arena;

typedef struct MatrixF {
    uint32_t ncols;
    uint32_t nrows;
    long double *values;
} MatrixF;

typedef struct MatrixI {
    uint32_t ncols;
    uint32_t nrows;
    uint32_t *values;
} MatrixI;

typedef struct VectorF {
    uint32_t nvalues;
    long double *values;
} VectorF;

typedef struct VectorI {
    uint32_t nvalues;
    uint32_t *values;
} VectorI;

typedef struct State {
    uint32_t xcount;
    uint32_t kmin;
    uint32_t kmax;
    bool     apply_deviation;
    Arena   *arena;
    VectorF *xsorted;
    MatrixF *cost;
    MatrixI *splits;
    VectorF *xsum;
    VectorF *xsumsq;
} State;

typedef struct RowParams {
    uint32_t row;
    uint32_t imin;
    uint32_t imax;
    uint32_t istep;
} RowParams;

typedef struct {
    long double mean;
    long double variance;
} SegmentStats;

VALUE        rb_ckmeans_sorted_group_sizes(VALUE self);

Arena       *arena_create(uint32_t);
void        *arena_alloc(Arena*, uint32_t);
void         arena_destroy(Arena*);

MatrixF     *matrix_create_f(Arena*, uint32_t, uint32_t);
MatrixI     *matrix_create_i(Arena*, uint32_t, uint32_t);
void         matrix_set_f(MatrixF*, uint32_t, uint32_t, long double value);
long double  matrix_get_f(MatrixF*, uint32_t, uint32_t);
void         matrix_inspect_f(MatrixF*);
void         matrix_set_i(MatrixI*, uint32_t, uint32_t, uint32_t value);
uint32_t     matrix_get_i(MatrixI*, uint32_t, uint32_t);
void         matrix_inspect_i(MatrixI*);

VectorF     *vector_create_f(Arena*, uint32_t);
void         vector_set_f(VectorF*, uint32_t offset, long double value);
long double  vector_get_f(VectorF*, uint32_t offset);
long double  vector_get_diff_f(VectorF*, uint32_t, uint32_t);
void         vector_inspect_f(VectorF*);
VectorI     *vector_create_i(Arena*, uint32_t);
VectorI     *vector_dup_i(VectorI*, Arena*);
void         vector_set_i(VectorI*, uint32_t offset, uint32_t value);
uint32_t     vector_get_i(VectorI*, uint32_t offset);
void         vector_downsize_i(VectorI*, uint32_t);
void         vector_inspect_i(VectorI*);

long double  dissimilarity(uint32_t, uint32_t, VectorF*, VectorF*);
void         fill_row(State, uint32_t, uint32_t, uint32_t);
void         smawk(State, RowParams, VectorI*);
void         find_min_from_candidates(State, RowParams, VectorI*);
VectorI     *prune_candidates(State, RowParams, VectorI*);
void         fill_even_positions(State, RowParams, VectorI*);
SegmentStats shifted_data_variance(VectorF*, uint32_t, uint32_t);
VectorI     *backtrack_sizes(State, uint32_t);
uint32_t     find_koptimal(State);

void Init_extensions(void) {
    VALUE ckmeans_module = rb_const_get(rb_cObject, rb_intern("Ckmeans"));
    VALUE clusterer_class = rb_const_get(ckmeans_module, rb_intern("Clusterer"));

    rb_define_private_method(clusterer_class, "sorted_group_sizes", rb_ckmeans_sorted_group_sizes, 0);
}

# define ARENA_MIN_CAPACITY 1024
# define ALLOCATION_FACTOR 20
# define PIx2 (M_PI * 2.0)

VALUE rb_ckmeans_sorted_group_sizes(VALUE self) {
    VALUE rb_xcount              = rb_ivar_get(self, rb_intern("@xcount"));
    VALUE rb_kmin                = rb_ivar_get(self, rb_intern("@kmin"));
    VALUE rb_kmax                = rb_ivar_get(self, rb_intern("@kmax"));
    VALUE rb_xsorted             = rb_ivar_get(self, rb_intern("@xsorted"));
    VALUE rb_apply_bic_deviation = rb_ivar_get(self, rb_intern("@apply_bic_deviation"));
    uint32_t xcount              = NUM2UINT(rb_xcount);
    uint32_t kmin                = NUM2UINT(rb_kmin);
    uint32_t kmax                = NUM2UINT(rb_kmax);
    bool apply_deviation         = RTEST(rb_apply_bic_deviation);
    Arena *arena                 = arena_create(sizeof(int) * xcount * kmax * ALLOCATION_FACTOR);

    if (arena == NULL) rb_raise(rb_eNoMemError, "Arena Memory Allocation Failed");

    MatrixF *cost    = matrix_create_f(arena, kmax, xcount);
    MatrixI *splits  = matrix_create_i(arena, kmax, xcount);
    VectorF *xsorted = vector_create_f(arena, xcount);
    VectorF *xsum    = vector_create_f(arena, xcount);
    VectorF *xsumsq  = vector_create_f(arena, xcount);

    for (uint32_t i = 0; i < xcount; i++) {
        long double xi = NUM2DBL(rb_ary_entry(rb_xsorted, i));
        vector_set_f(xsorted, i, xi);
    }

    State state = {
        .arena           = arena,
        .xcount          = xcount,
        .kmin            = kmin,
        .kmax            = kmax,
        .apply_deviation = apply_deviation,
        .xsorted         = xsorted,
        .cost            = cost,
        .splits          = splits,
        .xsum            = xsum,
        .xsumsq          = xsumsq
    };


    long double shift        = vector_get_f(xsorted, xcount / 2);
    long double diff_initial = vector_get_f(xsorted, 0) - shift;

    vector_set_f(xsum, 0, diff_initial);
    vector_set_f(xsumsq, 0, diff_initial * diff_initial);

    for (uint32_t i = 1; i < xcount; i++) {
        long double xi          = vector_get_f(xsorted, i);
        long double xsum_prev   = vector_get_f(xsum, i - 1);
        long double xsumsq_prev = vector_get_f(xsumsq, i - 1);
        long double diff        = xi - shift;

        vector_set_f(xsum, i, xsum_prev + diff);
        vector_set_f(xsumsq, i, xsumsq_prev + diff * diff);
        matrix_set_f(cost, 0, i, dissimilarity(0, i, xsum, xsumsq));
        matrix_set_i(splits, 0, i, 0);
    }

    for (uint32_t q = 1; q <= kmax - 1; q++) {
        uint32_t imin = (q < kmax - 1) ? ((q > 1) ? q : 1) : xcount - 1;
        fill_row(state, q, imin, xcount - 1);
    }

    uint32_t koptimal = find_koptimal(state);

    VectorI *sizes = backtrack_sizes(state, koptimal);

    /* printf("XSORTED \t"); vector_inspect_f(xsorted); */
    /* printf("K OPTIMAL: %lld\n", koptimal); */
    /* printf("SIZES \t"); vector_inspect_i(sizes); */
    /* printf("FINAL COST\n"); matrix_inspect_f(cost); */
    /* printf("FINAL SPLITS\n"); matrix_inspect_i(splits); */

    VALUE response = rb_ary_new2(sizes->nvalues);
    for (uint32_t i = 0; i < sizes->nvalues; i++) {
        VALUE size = LONG2NUM(vector_get_i(sizes, i));
        rb_ary_store(response, i, size);
    }

    arena_destroy(arena);

    return response;
}

uint32_t find_koptimal(State state)
{
    uint32_t kmin           = state.kmin;
    uint32_t kmax           = state.kmax;
    uint32_t xcount         = state.xcount;
    uint32_t kopt           = kmin;
    uint32_t xindex_max     = state.xcount - 1;
    VectorF *xsorted       = state.xsorted;
    long double x0         = vector_get_f(xsorted, 0);
    long double xn         = vector_get_f(xsorted, xindex_max);
    long double max_bic    = 0.0;
    long double adjustment = state.apply_deviation ? 0.0 : 1.0;

    for (uint32_t k = kmin; k <= kmax; k++) {
        uint32_t index_right, index_left = 0;
        long double bin_left, bin_right, loglikelihood = 0.0;
        VectorI *sizes = backtrack_sizes(state, k);

        for (uint32_t kb = 0; kb < k; kb++) {
            uint32_t npoints   = vector_get_i(sizes, kb);
            index_right        = index_left + npoints - 1;
            long double xleft  = vector_get_f(xsorted, index_left);
            long double xright = vector_get_f(xsorted, index_right);
            bin_left           = xleft;
            bin_right          = xright;

            if (xleft == xright) {
                bin_left  = index_left == 0
                    ? x0
                    : (vector_get_f(xsorted, index_left - 1) + xleft) / 2;
                bin_right = index_right < xindex_max
                    ? (xright + vector_get_f(xsorted, index_right + 1)) / 2
                    : xn;
            }

            long double bin_width = bin_right - bin_left;
            SegmentStats stats    = shifted_data_variance(xsorted, index_left, index_right);
            long double mean      = stats.mean;
            long double variance  = stats.variance;

            if (variance > 0) {
                for (uint32_t i = index_left; i <= index_right; i++) {
                    long double xi = vector_get_f(xsorted, i);
                    loglikelihood += -(xi - mean) * (xi - mean) / (2.0 * variance);
                }
                loglikelihood += npoints * (
                    (log(npoints / (long double) xcount) * adjustment) -
                    (0.5 * log(PIx2 * variance))
                );
            } else {
                loglikelihood += npoints * log(1.0 / bin_width / xcount);
            }

            index_left = index_right + 1;
        }

        long double bic = (2.0 * loglikelihood) - (((3 * k) - 1) * log((long double) xcount));

        if (k == kmin) {
            max_bic = bic;
            kopt = kmin;
        } else if (bic > max_bic) {
            max_bic = bic;
            kopt = k;
        }
    }

    return kopt;
}

VectorI *backtrack_sizes(State state, uint32_t k)
{
    MatrixI *splits = state.splits;
    VectorI *sizes  = vector_create_i(state.arena, k);
    uint32_t xcount = state.xcount;
    uint32_t right  = xcount - 1;
    uint32_t left   = 0;

    // Common case works with `i` remaining unsigned and unconditional assignment of the next `left` and `right`
    for (uint32_t i = k - 1; i > 0; i--, right = left - 1) {
        left = matrix_get_i(splits, i, right);
        vector_set_i(sizes, i, right - left + 1);
    }
    // Special case outside of the loop removing the need for conditionals
    left = matrix_get_i(splits, 0, right);
    vector_set_i(sizes, 0, right - left + 1);

    return sizes;
}

SegmentStats shifted_data_variance(VectorF *xsorted, uint32_t left, uint32_t right)
{
    const uint32_t n    = right - left + 1;
    long double sum    = 0.0;
    long double sumsq  = 0.0;
    SegmentStats stats = { .mean = 0.0, .variance = 0.0 };

    if (right >= left) {
        const long double median = vector_get_f(xsorted, (left + right) / 2);

        for (uint32_t i = left; i <= right; i++) {
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

void fill_row(State state, uint32_t q, uint32_t imin, uint32_t imax) {
    uint32_t size = imax - q + 1;
    VectorI *split_candidates = vector_create_i(state.arena, size);
    for (uint32_t i = 0; i < size; i++) {
        vector_set_i(split_candidates, i, q + i);
    }
    RowParams rparams = { .row = q, .imin = imin, .imax = imax, .istep = 1 };
    smawk(state, rparams, split_candidates);
}

void smawk(State state, RowParams rparams, VectorI *split_candidates) {
    const uint32_t imin  = rparams.imin;
    const uint32_t imax  = rparams.imax;
    const uint32_t istep = rparams.istep;

    if ((imax - imin) <= (0 * istep)) {
        find_min_from_candidates(state, rparams, split_candidates);
    } else {
        VectorI *odd_candidates = prune_candidates(state, rparams, split_candidates);
        /* printf("PRUNED\t"); vector_inspect_i(odd_candidates); */
        uint32_t istepx2         = istep * 2;
        uint32_t imin_odd        = imin + istep;
        uint32_t imax_odd        = imin_odd + ((imax - imin_odd) / istepx2 * istepx2);
        RowParams rparams_odd   = { .row = rparams.row, .imin = imin_odd, .imax = imax_odd, .istep = istepx2 };

        smawk(state, rparams_odd, odd_candidates);
        fill_even_positions(state, rparams, split_candidates);
    }
}

void fill_even_positions(State state, RowParams rparams, VectorI *split_candidates)
{
    uint32_t row     = rparams.row;
    uint32_t imin    = rparams.imin;
    uint32_t imax    = rparams.imax;
    uint32_t istep   = rparams.istep;
    uint32_t n       = split_candidates->nvalues;
    uint32_t istepx2 = istep * 2;
    uint32_t jl      = vector_get_i(split_candidates, 0);
    VectorF *xsum   = state.xsum;
    VectorF *xsumsq = state.xsumsq;
    MatrixI *splits = state.splits;

    for (uint32_t i = imin, r = 0; i <= imax; i += istepx2) {
        while (vector_get_i(split_candidates, r) < jl) r++;

        uint32_t rcandidate     = vector_get_i(split_candidates, r);
        uint32_t cost_base_row = row - 1;
        uint32_t cost_base_col = rcandidate - 1;
        long double cost      =
            matrix_get_f(state.cost, cost_base_row, cost_base_col) + dissimilarity(rcandidate, i, xsum, xsumsq);

        matrix_set_f(state.cost, row, i, cost);
        matrix_set_i(state.splits, row, i, rcandidate);

        uint32_t jh         =
            (i + istep) <= imax
            ? matrix_get_i(splits, row, i + istep)
            : vector_get_i(split_candidates, n - 1);

        uint32_t jmax       = jh < i ? jh : i;
        long double sjimin = dissimilarity(jmax, i, xsum, xsumsq);

        for (++r; r < n && vector_get_i(split_candidates, r) <= jmax; r++) {
            uint32_t jabs = vector_get_i(split_candidates, r);

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
    const uint32_t row     = rparams.row;
    const uint32_t imin    = rparams.imin;
    const uint32_t imax    = rparams.imax;
    const uint32_t istep   = rparams.istep;
    MatrixF *const cost   = state.cost;
    MatrixI *const splits = state.splits;

    uint32_t optimal_split_idx_prev = 0;

    for (uint32_t i = imin; i <= imax; i += istep)
    {
        const uint32_t optimal_split_idx = optimal_split_idx_prev;
        const uint32_t optimal_split     = vector_get_i(split_candidates, optimal_split_idx);
        const uint32_t cost_prev         = matrix_get_f(cost, row - 1, optimal_split - 1);
        const long double added_cost    = dissimilarity(optimal_split, i, state.xsum, state.xsumsq);

        matrix_set_f(cost, row, i, cost_prev + added_cost);
        matrix_set_i(splits, row, i, optimal_split);

        for (uint32_t r = optimal_split_idx + 1; r < split_candidates->nvalues; r++)
        {
            uint32_t split = vector_get_i(split_candidates, r);

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
    uint32_t n = ((rparams.imax - rparams.imin) / rparams.istep) + 1;
    uint32_t m = split_candidates->nvalues;

    if (n >= m) return split_candidates;

    int32_t left    = -1;
    uint32_t right  = 0;
    VectorI *pruned = vector_dup_i(split_candidates, state.arena);

    while (m > n)
    {
        uint32_t p         = left + 1;
        uint32_t i         = rparams.imin + p * rparams.istep;
        uint32_t j         = vector_get_i(pruned, right);
        uint32_t jnext     = vector_get_i(pruned, right + 1);
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

    for (uint32_t i = left + 1; i < m; i++) {
        /* TODO: extract `vector_setcpy_T` */
        vector_set_i(pruned, i, vector_get_i(pruned, right++));
    }

    vector_downsize_i(pruned, m);

    return pruned;
}

long double dissimilarity(uint32_t j, uint32_t i, VectorF *xsum, VectorF *xsumsq) {
    long double sji = 0.0;

    if (j >= i) return sji;

    if (j > 0) {
        long double segment_diff = vector_get_diff_f(xsum, i, j - 1);
        uint32_t segment_size    = i - j + 1;
        sji                      = vector_get_diff_f(xsumsq, i, j - 1) - (segment_diff * segment_diff / segment_size);
    } else {
        long double xsumi = vector_get_f(xsum, i);
        sji               = vector_get_f(xsumsq, i) - (xsumi * xsumi / (i + 1));
    }

    return (sji > 0) ? sji : 0.0;
}

VectorF *vector_create_f(Arena *arena, uint32_t nvalues) {
    VectorF *v;

    v          = arena_alloc(arena, sizeof(*v));
    v->values  = arena_alloc(arena, sizeof(*(v->values)) * nvalues);
    v->nvalues = nvalues;

    return v;
}

VectorI *vector_create_i(Arena *arena, uint32_t nvalues) {
    VectorI *v;

    v          = arena_alloc(arena, sizeof(*v));
    v->values  = arena_alloc(arena, sizeof(*(v->values)) * nvalues);
    v->nvalues = nvalues;

    return v;
}

VectorI *vector_dup_i(VectorI *v, Arena *arena)
{
    VectorI *vdup = vector_create_i(arena, v->nvalues);

    memcpy(vdup->values, v->values, sizeof(*(v->values)) * v->nvalues);

    return vdup;
}

void vector_set_f(VectorF *v, uint32_t offset, long double value) {
    *(v->values + offset) = value;
}

void vector_set_i(VectorI *v, uint32_t offset, uint32_t value) {
    *(v->values + offset) = value;
}

uint32_t vector_get_i(VectorI *v, uint32_t offset) {
    return *(v->values + offset);
}

void vector_downsize_i(VectorI *v, uint32_t new_size) {
    v->nvalues = new_size;
}

void vector_inspect_i(VectorI *v) {
    for (uint32_t i = 0; i < v->nvalues - 1; i++)
        printf("%u, ", vector_get_i(v, i));
    printf("%u\n", vector_get_i(v, v->nvalues - 1));
}

long double vector_get_f(VectorF *v, uint32_t offset) {
    return *(v->values + offset);
}

long double vector_get_diff_f(VectorF *v, uint32_t i, uint32_t j) {
    return *(v->values + i) - *(v->values + j);
}

void vector_inspect_f(VectorF *v) {
    for (uint32_t i = 0; i < v->nvalues - 1; i++)
        printf("%Lf, ", vector_get_f(v, i));
    printf("%Lf\n", vector_get_f(v, v->nvalues - 1));
}

MatrixF *matrix_create_f(Arena *arena, uint32_t nrows, uint32_t ncols) {
    MatrixF *m;

    m         = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * ncols * nrows);
    m->ncols  = ncols;
    m->nrows  = nrows;

    return m;
}

MatrixI *matrix_create_i(Arena *arena, uint32_t nrows, uint32_t ncols) {
    MatrixI *m;

    m         = arena_alloc(arena, sizeof(*m));
    m->values = arena_alloc(arena, sizeof(*(m->values)) * ncols * nrows);
    m->ncols  = ncols;
    m->nrows  = nrows;

    return m;
}

void matrix_set_f(MatrixF *m, uint32_t i, uint32_t j, long double value) {
    uint32_t offset = i * m->ncols + j;
    *(m->values + offset) = value;
}

long double matrix_get_f(MatrixF *m, uint32_t i, uint32_t j) {
    uint32_t offset = i * m->ncols + j;
    return *(m->values + offset);
}

void matrix_inspect_f(MatrixF *m) {
    for (uint32_t i = 0; i < m->nrows; i++) {
        for (uint32_t j = 0; j < m->ncols - 1; j++) {
            long double value = matrix_get_f(m, i, j);

            printf("%Lf, ", value);
        }
        printf("%Lf\n", matrix_get_f(m, i, m->ncols - 1));
    }
}

void matrix_inspect_i(MatrixI *m) {
    for (uint32_t i = 0; i < m->nrows; i++) {
        for (uint32_t j = 0; j < m->ncols - 1; j++)
            printf("%u, ", matrix_get_i(m, i, j));
        printf("%u\n", matrix_get_i(m, i, m->ncols - 1));
    }
}

void matrix_set_i(MatrixI *m, uint32_t i, uint32_t j, uint32_t value) {
    uint32_t offset = i * m->ncols + j;
    *(m->values + offset) = value;
}

uint32_t matrix_get_i(MatrixI *m, uint32_t i, uint32_t j) {
    uint32_t offset = i * m->ncols + j;
    return *(m->values + offset);
}

Arena *arena_create(uint32_t capacity) {
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
        printf("Failed to allocate arena\n");
        free(arena);
        return NULL;
    }

    arena->capacity = capacity;
    arena->offset   = 0;

    /* printf("[Arena Created] Capacity: %u, offset: %u\n", arena->capacity, arena->offset); */

    return arena;
}

void *arena_alloc(Arena *arena, uint32_t size) {
    size = (size + 7) & ~7;

    if (arena->offset + size > arena->capacity) {
        rb_raise(rb_eNoMemError, "Arena Insufficient Capacity");
        return NULL;
    }

    void *ptr = arena->buffer + arena->offset;
    arena->offset += size;

    return ptr;
}

void arena_destroy(Arena *arena) {
    /* printf("[Arena Destroy] Capacity: %u, offset: %u, left: %u\n", arena->capacity, arena->offset, arena->capacity - arena->offset); */
    free(arena->buffer);
    free(arena);
}
