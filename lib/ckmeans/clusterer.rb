# frozen_string_literal: true

module Ckmeans
  # Optimal k-means clustering for univariate (1D) data using dynamic programming.
  # Minimizes within-cluster sum of squared distances (L2 norm).
  class Clusterer
    # Creates a new Ckmeans clusterer.
    #
    # @param entries [Array<Numeric>] The data points to cluster
    # @param kmin [Integer] Minimum number of clusters to consider
    # @param kmax [Integer] Maximum number of clusters to consider (defaults to kmin for fixed K)
    # @param kestimate [Symbol] Method for estimating optimal K:
    #   - :fast   - Quick heuristic using implicit Gaussian assumption (best for large datasets)
    #   - :stable - Model-based estimation using Gaussian Mixture Model (better for duplicates/edge cases)
    #   - :gmm    - Alias for :stable (Gaussian Mixture Model)
    #
    # @example Fixed number of clusters
    #   Ckmeans::Clusterer.new([1, 2, 3, 100, 101], 2).clusters
    #   # => [[1, 2, 3], [100, 101]]
    #
    # @example Automatic K selection with stable estimation
    #   Ckmeans::Clusterer.new([1, 1, 1, 5, 5, 5, 10, 10, 10], 1, 5, :stable).clusters
    def initialize(entries, kmin, kmax = kmin, kestimate = :fast)
      @xcount = entries.size

      raise ArgumentError, "Minimum cluster count is bigger than element count" if kmin > @xcount
      raise ArgumentError, "Maximum cluster count is bigger than element count" if kmax > @xcount

      @kmin             = kmin
      @unique_xcount    = entries.uniq.size
      @kmax             = [@unique_xcount, kmax].min
      @xsorted_original = entries.sort
      @xsorted          = @xsorted_original.map(&:to_f)
      @use_gmm          = %i[gmm stable].include?(kestimate)
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
