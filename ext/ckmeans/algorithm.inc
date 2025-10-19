/* SMAWK algorithm implementation template
 *
 * This file is designed to be included multiple times with different DISSIM_SUFFIX and DISSIM macro definitions
 * to generate L1 and L2 specific versions of the clustering algorithm.
 *
 * Before including this file, define:
 *   DISSIM_SUFFIX              - suffix for function names (e.g., l1, l2)
 *   DISSIM(j, i, xsum, xsumsq) - macro that calls the dissimilarity function
 *
 * Example:
 *   #define DISSIM_SUFFIX l2
 *   #define DISSIM(j, i, xsum, xsumsq) dissimilarity_l2(j, i, xsum, xsumsq)
 *   #include "algorithm.inc"
 *   #undef DISSIM
 *   #undef DISSIM_SUFFIX
 */

#ifndef DISSIM_SUFFIX
#error "DISSIM_SUFFIX must be defined before including algorithm.inc"
#endif

#ifndef DISSIM
#error "DISSIM must be defined before including algorithm.inc"
#endif

/* Helper macros for token pasting */
#define CONCAT_IMPL(a, b) a##_##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define FUNC_NAME(name) CONCAT(name, DISSIM_SUFFIX)

static inline void FUNC_NAME(fill_even_positions)(State state, RowParams rparams, VectorI *split_candidates)
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

    for (uint32_t i = imin, r = 0; i <= imax; i += istepx2) {
        while (vector_get_i(split_candidates, r) < jl) r++;

        uint32_t rcandidate    = vector_get_i(split_candidates, r);
        uint32_t cost_base_row = row - 1;
        uint32_t cost_base_col = rcandidate - 1;
        LDouble cost           = matrix_get_f(state.cost, cost_base_row, cost_base_col) 
                               + DISSIM(rcandidate, i, xsum, xsumsq);

        matrix_set_f(state.cost, row, i, cost);
        matrix_set_i(state.splits, row, i, rcandidate);

        uint32_t jh = (i + istep) <= imax
            ? matrix_get_i(splits, row, i + istep)
            : vector_get_i(split_candidates, n - 1);

        uint32_t jmax  = jh < i ? jh : i;
        LDouble sjimin = DISSIM(jmax, i, xsum, xsumsq);

        for (++r; r < n && vector_get_i(split_candidates, r) <= jmax; r++) {
            uint32_t jabs = vector_get_i(split_candidates, r);

            if (jabs > i) break;
            if (jabs < matrix_get_i(splits, row - 1, i)) continue;

            LDouble cost_base = matrix_get_f(state.cost, row - 1, jabs  - 1);
            LDouble sj        = cost_base + DISSIM(jabs, i, xsum, xsumsq);
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

static inline void FUNC_NAME(find_min_from_candidates)(State state, RowParams rparams, VectorI *split_candidates)
{
    const uint32_t row     = rparams.row;
    const uint32_t imin    = rparams.imin;
    const uint32_t imax    = rparams.imax;
    const uint32_t istep   = rparams.istep;
    MatrixF *const cost    = state.cost;
    MatrixI *const splits  = state.splits;

    uint32_t optimal_split_idx_prev = 0;

    for (uint32_t i = imin; i <= imax; i += istep)
    {
        const uint32_t optimal_split_idx = optimal_split_idx_prev;
        const uint32_t optimal_split     = vector_get_i(split_candidates, optimal_split_idx);
        const uint32_t cost_prev         = matrix_get_f(cost, row - 1, optimal_split - 1);
        const LDouble added_cost         = DISSIM(optimal_split, i, state.xsum, state.xsumsq);

        matrix_set_f(cost, row, i, cost_prev + added_cost);
        matrix_set_i(splits, row, i, optimal_split);

        for (uint32_t r = optimal_split_idx + 1; r < split_candidates->size; r++)
        {
            uint32_t split = vector_get_i(split_candidates, r);

            if (split < matrix_get_i(splits, row - 1, i)) continue;
            if (split > i) break;

            LDouble split_cost = matrix_get_f(cost, row - 1, split - 1) + DISSIM(split, i, state.xsum, state.xsumsq);

            if (split_cost > matrix_get_f(cost, row, i)) continue;

            matrix_set_f(cost, row, i, split_cost);
            matrix_set_i(splits, row, i, split);
            optimal_split_idx_prev = r;
        }
    }
}

static inline VectorI *FUNC_NAME(prune_candidates)(State state, RowParams rparams, VectorI *split_candidates)
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

    while (m > n)
    {
        uint32_t i     = imin + left * istep;
        uint32_t j     = vector_get_i(pruned, right);
        uint32_t jnext = vector_get_i(pruned, right + 1);
        LDouble sl     = matrix_get_f(state.cost, row - 1, j - 1) + DISSIM(j, i, state.xsum, state.xsumsq);
        LDouble snext  = matrix_get_f(state.cost, row - 1, jnext - 1) + DISSIM(jnext, i, state.xsum, state.xsumsq);

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

void FUNC_NAME(smawk)(State state, RowParams rparams, VectorI *split_candidates)
{
    const uint32_t imin  = rparams.imin;
    const uint32_t imax  = rparams.imax;
    const uint32_t istep = rparams.istep;

    if ((imax - imin) <= (0 * istep)) {
        FUNC_NAME(find_min_from_candidates)(state, rparams, split_candidates);
    } else {
        VectorI *odd_candidates = FUNC_NAME(prune_candidates)(state, rparams, split_candidates);
        uint32_t istepx2        = istep * 2;
        uint32_t imin_odd       = imin + istep;
        uint32_t imax_odd       = imin_odd + ((imax - imin_odd) / istepx2 * istepx2);
        RowParams rparams_odd   = { .row = rparams.row, .imin = imin_odd, .imax = imax_odd, .istep = istepx2 };

        FUNC_NAME(smawk)(state, rparams_odd, odd_candidates);
        FUNC_NAME(fill_even_positions)(state, rparams, split_candidates);
    }
}

void FUNC_NAME(fill_row)(State state, uint32_t q, uint32_t imin, uint32_t imax)
{
    uint32_t size = imax - q + 1;
    VectorI *split_candidates = vector_create_i(state.arena, size);
    for (uint32_t i = 0; i < size; i++) {
        vector_set_i(split_candidates, i, q + i);
    }
    RowParams rparams = { .row = q, .imin = imin, .imax = imax, .istep = 1 };
    FUNC_NAME(smawk)(state, rparams, split_candidates);
}

VALUE FUNC_NAME(rb_sorted_group_sizes)(VALUE self)
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
        .xsumsq  = xsumsq
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
        matrix_set_f(cost, 0, i, DISSIM(0, i, xsum, xsumsq));
        matrix_set_i(splits, 0, i, 0);
    }

    for (uint32_t q = 1; q <= kmax - 1; q++) {
        uint32_t imin = (q < kmax - 1) ? ((q > 1) ? q : 1) : xcount - 1;
        FUNC_NAME(fill_row)(state, q, imin, xcount - 1);
    }

    uint32_t koptimal = use_gmm ? find_koptimal_gmm(state) : find_koptimal_fast(state);

    VectorI *sizes = vector_create_i(arena, koptimal);
    backtrack_sizes(state, sizes, koptimal);

    VALUE response = rb_ary_new2(sizes->size);
    for (uint32_t i = 0; i < sizes->size; i++) {
        VALUE size = LONG2NUM(vector_get_i(sizes, i));
        rb_ary_store(response, i, size);
    }

    arena_destroy(arena);

    return response;
}

#undef CONCAT_IMPL
#undef CONCAT
#undef FUNC_NAME
