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

typedef LDouble (FnDissim)(uint32_t, uint32_t, VectorF*, VectorF*);

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
    FnDissim *dissim;
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
VALUE rb_sorted_group_sizes(VALUE self, FnDissim*);

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

LDouble      dissimilarity_l2(uint32_t, uint32_t, VectorF*, VectorF*);
LDouble      dissimilarity_l1(uint32_t, uint32_t, VectorF*, VectorF*);
void         fill_row(State, uint32_t, uint32_t, uint32_t);
void         smawk(State, RowParams, VectorI*);
void         find_min_from_candidates(State, RowParams, VectorI*);
VectorI      *prune_candidates(State, RowParams, VectorI*);
void         fill_even_positions(State, RowParams, VectorI*);
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

VALUE rb_ckmeans_sorted_group_sizes(VALUE self)
{
    return rb_sorted_group_sizes(self, dissimilarity_l2);
}

VALUE rb_ckmedian_sorted_group_sizes(VALUE self)
{
    return rb_sorted_group_sizes(self, dissimilarity_l1);
}

VALUE rb_sorted_group_sizes(VALUE self, FnDissim *criteria)
{
    uint32_t xcount  = NUM2UINT(rb_iv_get(self, "@xcount"));
    uint32_t kmin    = NUM2UINT(rb_iv_get(self, "@kmin"));
    uint32_t kmax    = NUM2UINT(rb_iv_get(self, "@kmax"));
    bool use_gmm     = RTEST(rb_iv_get(self, "@use_gmm"));
    VALUE rb_xsorted = rb_iv_get(self, "@xsorted");
    size_t capacity  = sizeof(LDouble) * (xcount + 2) * (kmax + 2) * ALLOCATION_FACTOR + ARENA_MIN_CAPACITY;
    Arena *arena     = arena_create(capacity);

    if (arena == NULL) rb_raise(rb_eNoMemError, "Arena Memory Allocation Failed");

    MatrixF *cost    = matrix_create_f(arena, kmax, xcount);
    MatrixI *splits  = matrix_create_i(arena, kmax, xcount);
    VectorF *xsorted = vector_create_f(arena, xcount);
    VectorF *xsum    = vector_create_f(arena, xcount);
    VectorF *xsumsq  = vector_create_f(arena, xcount);

    for (uint32_t i = 0; i < xcount; i++) {
        LDouble xi = NUM2DBL(rb_ary_entry(rb_xsorted, i));
        vector_set_f(xsorted, i, xi);
    }

    State state = {
        .arena   = arena,
        .xcount  = xcount,
        .kmin    = kmin,
        .kmax    = kmax,
        .xsorted = xsorted,
        .cost    = cost,
        .splits  = splits,
        .xsum    = xsum,
        .xsumsq  = xsumsq,
        .dissim  = criteria
    };


    LDouble shift        = vector_get_f(xsorted, xcount / 2);
    LDouble diff_initial = vector_get_f(xsorted, 0) - shift;

    vector_set_f(xsum, 0, diff_initial);
    vector_set_f(xsumsq, 0, diff_initial * diff_initial);

    for (uint32_t i = 1; i < xcount; i++) {
        LDouble xi          = vector_get_f(xsorted, i);
        LDouble xsum_prev   = vector_get_f(xsum, i - 1);
        LDouble xsumsq_prev = vector_get_f(xsumsq, i - 1);
        LDouble diff        = xi - shift;

        vector_set_f(xsum, i, xsum_prev + diff);
        vector_set_f(xsumsq, i, xsumsq_prev + diff * diff);
        matrix_set_f(cost, 0, i, criteria(0, i, xsum, xsumsq));
        matrix_set_i(splits, 0, i, 0);
    }

    for (uint32_t q = 1; q <= kmax - 1; q++) {
        uint32_t imin = (q < kmax - 1) ? ((q > 1) ? q : 1) : xcount - 1;
        fill_row(state, q, imin, xcount - 1);
    }

    uint32_t koptimal = use_gmm ? find_koptimal_gmm(state) : find_koptimal_fast(state);

    VectorI *sizes = vector_create_i(arena, koptimal);
    backtrack_sizes(state, sizes, koptimal);

    /* printf("XSORTED \t"); vector_inspect_f(xsorted); */
    /* printf("K OPTIMAL: %lld\n", koptimal); */
    /* printf("SIZES \t"); vector_inspect_i(sizes); */
    /* printf("FINAL COST\n"); matrix_inspect_f(cost); */
    /* printf("FINAL SPLITS\n"); matrix_inspect_i(splits); */

    VALUE response = rb_ary_new2(sizes->size);
    for (uint32_t i = 0; i < sizes->size; i++) {
        VALUE size = LONG2NUM(vector_get_i(sizes, i));
        rb_ary_store(response, i, size);
    }

    arena_destroy(arena);

    return response;
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

void fill_row(State state, uint32_t q, uint32_t imin, uint32_t imax)
{
    uint32_t size = imax - q + 1;
    VectorI *split_candidates = vector_create_i(state.arena, size);
    for (uint32_t i = 0; i < size; i++) {
        vector_set_i(split_candidates, i, q + i);
    }
    RowParams rparams = { .row = q, .imin = imin, .imax = imax, .istep = 1 };
    smawk(state, rparams, split_candidates);
}

void smawk(State state, RowParams rparams, VectorI *split_candidates)
{
    const uint32_t imin  = rparams.imin;
    const uint32_t imax  = rparams.imax;
    const uint32_t istep = rparams.istep;

    if ((imax - imin) <= (0 * istep)) {
        find_min_from_candidates(state, rparams, split_candidates);
    } else {
        VectorI *odd_candidates = prune_candidates(state, rparams, split_candidates);
        /* printf("PRUNED\t"); vector_inspect_i(odd_candidates); */
        uint32_t istepx2        = istep * 2;
        uint32_t imin_odd       = imin + istep;
        uint32_t imax_odd       = imin_odd + ((imax - imin_odd) / istepx2 * istepx2);
        RowParams rparams_odd   = { .row = rparams.row, .imin = imin_odd, .imax = imax_odd, .istep = istepx2 };

        smawk(state, rparams_odd, odd_candidates);
        fill_even_positions(state, rparams, split_candidates);
    }
}

inline void fill_even_positions(State state, RowParams rparams, VectorI *split_candidates)
{
    uint32_t row     = rparams.row;
    uint32_t imin    = rparams.imin;
    uint32_t imax    = rparams.imax;
    uint32_t istep   = rparams.istep;
    uint32_t n       = split_candidates->size;
    uint32_t istepx2 = istep * 2;
    uint32_t jl      = vector_get_i(split_candidates, 0);
    VectorF *const xsum    = state.xsum;
    VectorF *const xsumsq  = state.xsumsq;
    MatrixI *const splits  = state.splits;
    FnDissim *const dissim = state.dissim;

    for (uint32_t i = imin, r = 0; i <= imax; i += istepx2) {
        while (vector_get_i(split_candidates, r) < jl) r++;

        uint32_t rcandidate    = vector_get_i(split_candidates, r);
        uint32_t cost_base_row = row - 1;
        uint32_t cost_base_col = rcandidate - 1;
        LDouble cost           =
            matrix_get_f(state.cost, cost_base_row, cost_base_col) + dissim(rcandidate, i, xsum, xsumsq);

        matrix_set_f(state.cost, row, i, cost);
        matrix_set_i(state.splits, row, i, rcandidate);

        uint32_t jh =
            (i + istep) <= imax
            ? matrix_get_i(splits, row, i + istep)
            : vector_get_i(split_candidates, n - 1);

        uint32_t jmax  = jh < i ? jh : i;
        LDouble sjimin = dissim(jmax, i, xsum, xsumsq);

        for (++r; r < n && vector_get_i(split_candidates, r) <= jmax; r++) {
            uint32_t jabs = vector_get_i(split_candidates, r);

            if (jabs > i) break;
            if (jabs < matrix_get_i(splits, row - 1, i)) continue;

            LDouble cost_base = matrix_get_f(state.cost, row - 1, jabs  - 1);
            LDouble sj        = cost_base + dissim(jabs, i, xsum, xsumsq);
            LDouble cost_prev = matrix_get_f(state.cost, row, i);

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

inline void find_min_from_candidates(State state, RowParams rparams, VectorI *split_candidates)
{
    const uint32_t row     = rparams.row;
    const uint32_t imin    = rparams.imin;
    const uint32_t imax    = rparams.imax;
    const uint32_t istep   = rparams.istep;
    MatrixF *const cost    = state.cost;
    MatrixI *const splits  = state.splits;
    FnDissim *const dissim = state.dissim;

    uint32_t optimal_split_idx_prev = 0;

    for (uint32_t i = imin; i <= imax; i += istep)
    {
        const uint32_t optimal_split_idx = optimal_split_idx_prev;
        const uint32_t optimal_split     = vector_get_i(split_candidates, optimal_split_idx);
        const uint32_t cost_prev         = matrix_get_f(cost, row - 1, optimal_split - 1);
        const LDouble added_cost         = dissim(optimal_split, i, state.xsum, state.xsumsq);

        matrix_set_f(cost, row, i, cost_prev + added_cost);
        matrix_set_i(splits, row, i, optimal_split);

        for (uint32_t r = optimal_split_idx + 1; r < split_candidates->size; r++)
        {
            uint32_t split = vector_get_i(split_candidates, r);

            if (split < matrix_get_i(splits, row - 1, i)) continue;
            if (split > i) break;

            LDouble split_cost =
                matrix_get_f(cost, row - 1, split - 1) + dissim(split, i, state.xsum, state.xsumsq);

            if (split_cost > matrix_get_f(cost, row, i)) continue;

            matrix_set_f(cost, row, i, split_cost);
            matrix_set_i(splits, row, i, split);
            optimal_split_idx_prev = r;
        }
    }
}

inline VectorI *prune_candidates(State state, RowParams rparams, VectorI *split_candidates)
{
    uint32_t imin  = rparams.imin;
    uint32_t row   = rparams.row;
    uint32_t istep = rparams.istep;
    uint32_t n     = ((rparams.imax - imin) / istep) + 1;
    uint32_t m     = split_candidates->size;

    if (n >= m) return split_candidates;

    uint32_t left   = 0;
    uint32_t right  = 0;
    VectorI *pruned = vector_dup_i(split_candidates, state.arena);
    FnDissim *const dissim = state.dissim;

    while (m > n)
    {
        uint32_t i     = imin + left * istep;
        uint32_t j     = vector_get_i(pruned, right);
        uint32_t jnext = vector_get_i(pruned, right + 1);
        LDouble sl     =
            matrix_get_f(state.cost, row - 1, j - 1) + dissim(j, i, state.xsum, state.xsumsq);
        LDouble snext  =
            matrix_get_f(state.cost, row - 1, jnext - 1) + dissim(jnext, i, state.xsum, state.xsumsq);

        if ((sl < snext) && (left < n - 1)) {
            vector_set_i(pruned, left, j);
            left++;
            right++;
        } else if ((sl < snext) && (left == n - 1)) {
            right++;
            m--;
            vector_set_i(pruned, right, j);
        } else {
            if (left > 0) {
                vector_set_i(pruned, right, vector_get_i(pruned, --left));
            } else {
                right++;
            }

            m--;
        }
    }

    for (uint32_t i = left; i < m; i++) {
        vector_set_i(pruned, i, vector_get_i(pruned, right++));
    }

    vector_downsize_i(pruned, m);

    return pruned;
}

/* L2 aka Euclidean aka Mean dissimilarity criteria */
inline LDouble dissimilarity_l2(uint32_t j, uint32_t i, VectorF *restrict xsum, VectorF *restrict xsumsq) {
    LDouble sji = 0.0;

    if (j >= i) return sji;

    if (j > 0) {
        LDouble segment_diff  = vector_get_diff_f(xsum, i, j - 1);
        uint32_t segment_size = i - j + 1;
        sji                   = vector_get_diff_f(xsumsq, i, j - 1) - (segment_diff * segment_diff / segment_size);
    } else {
        LDouble xsumi = vector_get_f(xsum, i);
        sji           = vector_get_f(xsumsq, i) - (xsumi * xsumi / (i + 1));
    }

    return (sji > 0) ? sji : 0.0;
}

/* L1 aka Manhattan aka Median dissimilarity criteria */
inline LDouble dissimilarity_l1(uint32_t j, uint32_t i, VectorF *restrict xsum, VectorF *restrict _xsumsq)
{
    LDouble sji = 0.0;

    if (j >= i) return sji;

    if (j > 0) {
        uint32_t median_idx = (i + j) >> 1;

        if (((i - j + 1) % 2) == 1) {
            sji =
                - vector_get_f(xsum, median_idx - 1)
                + vector_get_f(xsum, j - 1)
                + vector_get_f(xsum, i)
                - vector_get_f(xsum, median_idx);
        } else {
            sji =
                - vector_get_f(xsum, median_idx)
                + vector_get_f(xsum, j - 1)
                + vector_get_f(xsum, i)
                - vector_get_f(xsum, median_idx);
        }
    } else { // j == 0
        uint32_t median_idx = i >> 1;

        if (((i + 1) % 2) == 1) {
            sji =
                - vector_get_f(xsum, median_idx - 1)
                + vector_get_f(xsum, i)
                - vector_get_f(xsum, median_idx);
        } else {
            sji =
                - vector_get_f(xsum, median_idx)
                + vector_get_f(xsum, i)
                - vector_get_f(xsum, median_idx);
        }
    }

    return (sji < 0) ? 0.0 : sji;
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
