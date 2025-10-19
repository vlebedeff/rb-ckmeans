#ifndef DISSIMILARITY_H
#define DISSIMILARITY_H

/* L2 aka Euclidean aka Mean dissimilarity criteria */
static inline LDouble dissimilarity_l2(uint32_t j, uint32_t i, VectorF *restrict xsum, VectorF *restrict xsumsq)
{
    LDouble sji = 0.0;

    if (j >= i) return sji;

    if (j > 0) {
        LDouble segment_diff  = vector_get_diff_f(xsum, i, j - 1);
        uint32_t segment_size = i - j + 1;
        sji = vector_get_diff_f(xsumsq, i, j - 1) - (segment_diff * segment_diff / segment_size);
    } else {
        LDouble xsumi = vector_get_f(xsum, i);
        sji = vector_get_f(xsumsq, i) - (xsumi * xsumi / (i + 1));
    }

    return (sji > 0) ? sji : 0.0;
}

/* L1 aka Manhattan aka Median dissimilarity criteria */
static inline LDouble dissimilarity_l1(uint32_t j, uint32_t i, VectorF *restrict xsum, VectorF *restrict _xsumsq)
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

#endif /* DISSIMILARITY_H */
