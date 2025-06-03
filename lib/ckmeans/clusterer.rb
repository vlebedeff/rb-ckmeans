# frozen_string_literal: true

module Ckmeans
  class Clusterer # rubocop:disable Style/Documentation
    def initialize(entries, kmin, kmax = kmin, kestimate = :fast)
      @xcount = entries.size

      raise ArgumentError, "Minimum cluster count is bigger than element count" if kmin > @xcount
      raise ArgumentError, "Maximum cluster count is bigger than element count" if kmax > @xcount

      @kmin             = kmin
      @unique_xcount    = entries.uniq.size
      @kmax             = [@unique_xcount, kmax].min
      @xsorted_original = entries.sort
      @xsorted          = @xsorted_original.map(&:to_f)
      @use_gmm          = kestimate == :gmm
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
