#include <stdio.h>
#include <math.h>
#include <string.h>
#include "ruby.h"

typedef long double LDouble;

typedef struct Arena {
    size_t capacity;
    size_t offset;
    uint8_t  *buffer;
} Arena;

typedef struct MatrixF {
    uint32_t ncols;
    uint32_t nrows;
    LDouble *values;
} MatrixF;

typedef struct MatrixI {
    uint32_t ncols;
    uint32_t nrows;
    uint32_t *values;
} MatrixI;

typedef struct VectorF {
    uint32_t size;
    LDouble *values;
} VectorF;

typedef struct VectorI {
    uint32_t size;
    uint32_t *values;
} VectorI;

typedef struct State {
    uint32_t xcount;
    uint32_t kmin;
    uint32_t kmax;
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
    LDouble mean;
    LDouble variance;
} SegmentStats;

VALUE rb_ckmeans_sorted_group_sizes(VALUE self);
VALUE rb_ckmedian_sorted_group_sizes(VALUE self);

Arena *arena_create(size_t);
void  *arena_alloc(Arena*, size_t);
void  arena_destroy(Arena*);

MatrixF  *matrix_create_f(Arena*, uint32_t, uint32_t);
MatrixI  *matrix_create_i(Arena*, uint32_t, uint32_t);
void     matrix_set_f(MatrixF*, uint32_t, uint32_t, LDouble value);
LDouble  matrix_get_f(MatrixF*, uint32_t, uint32_t);
void     matrix_inspect_f(MatrixF*);
void     matrix_set_i(MatrixI*, uint32_t, uint32_t, uint32_t value);
uint32_t matrix_get_i(MatrixI*, uint32_t, uint32_t);
void     matrix_inspect_i(MatrixI*);

VectorF  *vector_create_f(Arena*, uint32_t);
void     vector_set_f(VectorF*, uint32_t offset, LDouble value);
LDouble  vector_get_f(VectorF*, uint32_t offset);
LDouble  vector_get_diff_f(VectorF*, uint32_t, uint32_t);
void     vector_inspect_f(VectorF*);
VectorI  *vector_create_i(Arena*, uint32_t);
VectorI  *vector_dup_i(VectorI*, Arena*);
void     vector_set_i(VectorI*, uint32_t offset, uint32_t value);
uint32_t vector_get_i(VectorI*, uint32_t offset);
void     vector_downsize_i(VectorI*, uint32_t);
void     vector_inspect_i(VectorI*);

SegmentStats shifted_data_variance(VectorF*, uint32_t, uint32_t);
VectorI      *backtrack_sizes(State, VectorI*, uint32_t);
uint32_t     find_koptimal_fast(State);
uint32_t     find_koptimal_gmm(State);

void Init_extensions(void) {
    VALUE ckmeans_module     = rb_const_get(rb_cObject, rb_intern("Ckmeans"));
    VALUE ckmedian_module    = rb_const_get(rb_cObject, rb_intern("Ckmedian"));
    VALUE ckmeans_clusterer  = rb_const_get(ckmeans_module, rb_intern("Clusterer"));
    VALUE ckmedian_clusterer = rb_const_get(ckmedian_module, rb_intern("Clusterer"));

    rb_define_private_method(ckmeans_clusterer, "sorted_group_sizes", rb_ckmeans_sorted_group_sizes, 0);
    rb_define_private_method(ckmedian_clusterer, "sorted_group_sizes", rb_ckmedian_sorted_group_sizes, 0);
}

# define ARENA_MIN_CAPACITY 100
# define ALLOCATION_FACTOR 3
# define PIx2 (M_PI * 2.0)

#include "dissimilarity.h"

/* L2-specific versions of all hot-path functions */
#define DISSIM_SUFFIX l2
#define DISSIM(j, i, xsum, xsumsq) dissimilarity_l2(j, i, xsum, xsumsq)
#include "algorithm.inc"
#undef DISSIM
#undef DISSIM_SUFFIX

/* L1-specific versions of all hot-path functions */
#define DISSIM_SUFFIX l1
#define DISSIM(j, i, xsum, xsumsq) dissimilarity_l1(j, i, xsum, xsumsq)
#include "algorithm.inc"
#undef DISSIM
#undef DISSIM_SUFFIX

VALUE rb_ckmeans_sorted_group_sizes(VALUE self)
{
    return rb_sorted_group_sizes_l2(self);
}
VALUE rb_ckmedian_sorted_group_sizes(VALUE self)
{
    return rb_sorted_group_sizes_l1(self);
}


uint32_t find_koptimal_fast(State state)
{
    uint32_t kmin       = state.kmin;
    uint32_t kmax       = state.kmax;
    uint32_t xcount     = state.xcount;
    uint32_t kopt       = kmin;
    uint32_t xindex_max = state.xcount - 1;
    VectorF *xsorted    = state.xsorted;
    LDouble x0          = vector_get_f(xsorted, 0);
    LDouble xn          = vector_get_f(xsorted, xindex_max);
    LDouble max_bic     = 0.0;
    LDouble xcount_log  = log((LDouble) xcount);

    VectorI *sizes = vector_create_i(state.arena, kmax);
    for (uint32_t k = kmin; k <= kmax; k++) {
        uint32_t index_right, index_left = 0;
        LDouble bin_left, bin_right, loglikelihood = 0.0;
        backtrack_sizes(state, sizes, k);

        for (uint32_t kb = 0; kb < k; kb++) {
            uint32_t npoints = vector_get_i(sizes, kb);
            index_right      = index_left + npoints - 1;
            LDouble xleft    = vector_get_f(xsorted, index_left);
            LDouble xright   = vector_get_f(xsorted, index_right);
            bin_left         = xleft;
            bin_right        = xright;

            if (xleft == xright) {
                bin_left  = index_left == 0
                    ? x0
                    : (vector_get_f(xsorted, index_left - 1) + xleft) / 2;
                bin_right = index_right < xindex_max
                    ? (xright + vector_get_f(xsorted, index_right + 1)) / 2
                    : xn;
            }

            LDouble bin_width  = bin_right - bin_left;
            SegmentStats stats = shifted_data_variance(xsorted, index_left, index_right);
            LDouble mean       = stats.mean;
            LDouble variance   = stats.variance;

            if (variance > 0) {
                for (uint32_t i = index_left; i <= index_right; i++) {
                    LDouble xi     = vector_get_f(xsorted, i);
                    loglikelihood += -(xi - mean) * (xi - mean) / (2.0 * variance);
                }
                loglikelihood += npoints * (
                    log(npoints / (LDouble) xcount) - (0.5 * log(PIx2 * variance))
                );
            } else {
                loglikelihood += npoints * log(1.0 / bin_width / xcount);
            }

            index_left = index_right + 1;
        }

        LDouble bic = (2.0 * loglikelihood) - (((3 * k) - 1) * xcount_log);

        if (k == kmin) {
            max_bic = bic;
            kopt    = kmin;
        } else if (bic > max_bic) {
            max_bic = bic;
            kopt    = k;
        }
    }

    return kopt;
}

uint32_t find_koptimal_gmm(State state)
{
    uint32_t kmin = state.kmin;
    uint32_t kmax = state.kmax;
    uint32_t xcount = state.xcount;

    if (kmin > kmax || xcount < 2) {
        return (kmin < kmax) ? kmin : kmax;
    }

    Arena *arena       = state.arena;
    VectorF *xsorted   = state.xsorted;
    uint32_t kopt      = kmin;
    LDouble max_bic    = 0.0;
    LDouble log_xcount = log((LDouble) xcount);
    VectorF *lambda    = vector_create_f(arena, kmax);
    VectorF *mu        = vector_create_f(arena, kmax);
    VectorF *sigma2    = vector_create_f(arena, kmax);
    VectorF *coeff     = vector_create_f(arena, kmax);
    VectorI *sizes     = vector_create_i(arena, kmax);

    for (uint32_t kouter = kmin; kouter <= kmax; ++kouter)
    {
        uint32_t ileft = 0;
        uint32_t iright;

        backtrack_sizes(state, sizes, kouter);

        for (uint32_t k = 0; k < kouter; ++k)
        {
            uint32_t size = vector_get_i(sizes, k);
            vector_set_f(lambda, k, size / (LDouble) xcount);
            iright = ileft + size - 1;
            SegmentStats stats = shifted_data_variance(xsorted, ileft, iright);

            vector_set_f(mu, k, stats.mean);
            vector_set_f(sigma2, k, stats.variance);

            if (stats.variance == 0 || size == 1) {
                LDouble dmin;

                if (ileft > 0 && iright < xcount - 1) {
                    LDouble left_diff = vector_get_diff_f(xsorted, ileft, ileft - 1);
                    LDouble right_diff = vector_get_diff_f(xsorted, iright + 1, iright);

                    dmin = (left_diff < right_diff) ? left_diff : right_diff;
                } else if (ileft > 0) {
                    dmin = vector_get_diff_f(xsorted, ileft, ileft - 1);
                } else {
                    dmin = vector_get_diff_f(xsorted, iright + 1, iright);
                }

                if (stats.variance == 0) vector_set_f(sigma2, k, dmin * dmin / 4.0 / 9.0);
                if (size == 1)  vector_set_f(sigma2, k, dmin * dmin);
            }

            LDouble lambda_k = vector_get_f(lambda, k);
            LDouble sigma2_k = vector_get_f(sigma2, k);
            vector_set_f(coeff, k, lambda_k / sqrt(PIx2 * sigma2_k));
            ileft = iright + 1;
        }

        LDouble loglikelihood = 0.0;

        for (uint32_t i = 0; i < xcount; ++i)
        {
            LDouble L  = 0.0;
            LDouble xi = vector_get_f(xsorted, i);

            for (uint32_t k = 0; k < kouter; ++k)
            {
                LDouble coeff_k   = vector_get_f(coeff, k);
                LDouble mu_k      = vector_get_f(mu, k);
                LDouble sigma2_k  = vector_get_f(sigma2, k);
                LDouble x_mu_diff = xi - mu_k;
                L                += coeff_k * exp(- x_mu_diff * x_mu_diff / (2.0 * sigma2_k));
            }
            loglikelihood += log(L);
        }

        LDouble bic = 2 * loglikelihood - (3 * kouter - 1) * log_xcount;

        if (kouter == kmin) {
            max_bic = bic;
            kopt = kmin;
        } else {
            if (bic > max_bic) {
                max_bic = bic;
                kopt = kouter;
            }
        }
    }
    return kopt;
}

VectorI *backtrack_sizes(State state, VectorI *sizes, uint32_t k)
{
    MatrixI *splits = state.splits;
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
    const uint32_t n   = right - left + 1;
    LDouble sum        = 0.0;
    LDouble sumsq      = 0.0;
    SegmentStats stats = { .mean = 0.0, .variance = 0.0 };

    if (right >= left) {
        const LDouble median = vector_get_f(xsorted, (left + right) / 2);

        for (uint32_t i = left; i <= right; i++) {
            const LDouble sumi = vector_get_f(xsorted, i) - median;

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

inline VectorF *vector_create_f(Arena *arena, uint32_t size) {
    VectorF *v;

    v         = arena_alloc(arena, sizeof(*v));
    v->values = arena_alloc(arena, sizeof(*(v->values)) * size);
    v->size   = size;

    return v;
}

inline VectorI *vector_create_i(Arena *arena, uint32_t size) {
    VectorI *v;

    v         = arena_alloc(arena, sizeof(*v));
    v->values = arena_alloc(arena, sizeof(*(v->values)) * size);
    v->size   = size;

    return v;
}

inline VectorI *vector_dup_i(VectorI *v, Arena *arena)
{
    VectorI *vdup = vector_create_i(arena, v->size);

    memcpy(vdup->values, v->values, sizeof(*(v->values)) * v->size);

    return vdup;
}

inline void vector_set_f(VectorF *v, uint32_t offset, LDouble value) {
    *(v->values + offset) = value;
}

inline void vector_set_i(VectorI *v, uint32_t offset, uint32_t value) {
    *(v->values + offset) = value;
}

inline uint32_t vector_get_i(VectorI *v, uint32_t offset) {
    return *(v->values + offset);
}

inline void vector_downsize_i(VectorI *v, uint32_t new_size) {
    v->size = new_size;
}

void vector_inspect_i(VectorI *v) {
    for (uint32_t i = 0; i < v->size - 1; i++)
        printf("%u, ", vector_get_i(v, i));
    printf("%u\n", vector_get_i(v, v->size - 1));
}

inline LDouble vector_get_f(VectorF *v, uint32_t offset) {
    return *(v->values + offset);
}

inline LDouble vector_get_diff_f(VectorF *v, uint32_t i, uint32_t j) {
    return *(v->values + i) - *(v->values + j);
}

void vector_inspect_f(VectorF *v) {
    for (uint32_t i = 0; i < v->size - 1; i++)
        printf("%Lf, ", vector_get_f(v, i));
    printf("%Lf\n", vector_get_f(v, v->size - 1));
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

inline void matrix_set_f(MatrixF *m, uint32_t i, uint32_t j, LDouble value) {
    uint32_t offset = i * m->ncols + j;
    *(m->values + offset) = value;
}

inline LDouble matrix_get_f(MatrixF *m, uint32_t i, uint32_t j) {
    uint32_t offset = i * m->ncols + j;
    return *(m->values + offset);
}

void matrix_inspect_f(MatrixF *m) {
    for (uint32_t i = 0; i < m->nrows; i++) {
        for (uint32_t j = 0; j < m->ncols - 1; j++) {
            LDouble value = matrix_get_f(m, i, j);

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

inline void matrix_set_i(MatrixI *m, uint32_t i, uint32_t j, uint32_t value) {
    uint32_t offset = i * m->ncols + j;
    *(m->values + offset) = value;
}

inline uint32_t matrix_get_i(MatrixI *m, uint32_t i, uint32_t j) {
    uint32_t offset = i * m->ncols + j;
    return *(m->values + offset);
}

Arena *arena_create(size_t capacity) {
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

void *arena_alloc(Arena *arena, size_t size) {
    size = (size + 0xf) & ~0xf;

    if (arena->offset + size > arena->capacity) {
        rb_raise(rb_eNoMemError, "Arena Insufficient Capacity");
        return NULL;
    }

    void *ptr = arena->buffer + arena->offset;
    arena->offset += size;

    return ptr;
}

void arena_destroy(Arena *arena) {
    /* double leftover = ((double) arena->capacity - arena->offset) / arena->capacity * 100; */
    /* printf("[Arena Destroy] Capacity: %zu, offset: %zu, left: %2.2f%%\n", arena->capacity, arena->offset, leftover); */
    free(arena->buffer);
    free(arena);
}
