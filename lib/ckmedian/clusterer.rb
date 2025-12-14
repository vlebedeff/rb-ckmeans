# frozen_string_literal: true

module Ckmedian
  # Optimal k-median clustering for univariate (1D) data using dynamic programming.
  # Minimizes within-cluster sum of absolute deviations (L1 norm).
  # More robust to outliers than k-means.
  class Clusterer
    # Creates a new Ckmedian clusterer.
    #
    # @param entries [Array<Numeric>] The data points to cluster
    # @param kmin [Integer] Minimum number of clusters to consider
    # @param kmax [Integer] Maximum number of clusters to consider (defaults to kmin for fixed K)
    # @param kestimate [Symbol] Method for estimating optimal K:
    #   - :fast   - Quick heuristic using implicit Gaussian assumption (best for large datasets)
    #   - :stable - Model-based estimation using Laplace Mixture Model (better for outliers/bursts)
    #   - :lmm    - Alias for :stable (Laplace Mixture Model)
    #
    # @example Fixed number of clusters
    #   Ckmedian::Clusterer.new([1, 2, 3, 100, 101], 2).clusters
    #   # => [[1, 2, 3], [100, 101]]
    #
    # @example Photo timeline clustering (robust to bursts and outliers)
    #   timestamps = photos.map(&:taken_at).map(&:to_i)
    #   Ckmedian::Clusterer.new(timestamps, 1, 20, :stable).clusters
    def initialize(entries, kmin, kmax = kmin, kestimate = :fast)
      @xcount = entries.size

      raise ArgumentError, "Minimum cluster count is bigger than element count" if kmin > @xcount
      raise ArgumentError, "Maximum cluster count is bigger than element count" if kmax > @xcount

      @kmin                  = kmin
      @unique_xcount         = entries.uniq.size
      @kmax                  = [@unique_xcount, kmax].min
      @xsorted_original      = entries.sort
      @xsorted               = @xsorted_original.map(&:to_f)
      @use_stable_estimation = %i[lmm stable].include?(kestimate)
    end

    def clusters
      @clusters ||=
        if @unique_xcount <= 1
          [@xsorted_original]
        else
          sorted_group_sizes.each_with_object([]) do |size, groups|
            groups << @xsorted_original.shift(size)
          end
        end
    end
  end
end
