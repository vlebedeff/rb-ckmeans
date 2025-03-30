# frozen_string_literal: true

module Ckmeans
  class Clusterer # rubocop:disable Style/Documentation, Metrics/ClassLength
    attr_reader :entry_count, :entries, :kmin, :kmax

    PI_DOUBLE = Math::PI * 2

    def initialize(entries, kmin, kmax = kmin)
      @entry_count = entries.size

      raise ArgumentError, "Minimum cluster count is bigger than element count" if kmin > @entry_count
      raise ArgumentError, "Maximum cluster count is bigger than element count" if kmax > @entry_count

      @kmin = kmin
      @unique_entry_count = entries.uniq.size
      @kmax = [@unique_entry_count, kmax].min
      @entries = entries.sort
      @entries_padded = [nil] + @entries
      @matrix_populated = false
    end

    def clusters # rubocop:disable Metrics/MethodLength
      @clusters ||=
        if @unique_entry_count <= 1
          [entries]
        else
          populate_matrix!
          results = []
          backtrack_from(kopt) do |k, left, right|
            results[k] = @entries_padded[left..right]
          end
          results.drop(1)
        end
    end

    private

    def distance
      @distance ||= Array.new(kmax + 1) { Array.new(entry_count + 1) { 0.0 } }
    end

    def backtrack
      @backtrack ||= Array.new(kmax + 1) { Array.new(entry_count + 1) { 0 } }
    end

    def populate_matrix! # rubocop:disable Metrics/AbcSize, Metrics/MethodLength, Metrics/PerceivedComplexity
      1.upto(kmax) do |i|
        distance[i][1] = 0.0
        backtrack[i][1] = 1
      end

      1.upto(kmax) do |k| # rubocop:disable Metrics/BlockLength
        mean_x1 = @entries_padded[1]
        distance_k = distance.at(k)
        distance_k_prev = distance.at(k - 1)
        backtrack1 = backtrack.at(1)
        backtrack_k = backtrack.at(k)

        [2, k].max.upto(entry_count) do |i| # rubocop:disable Metrics/BlockLength
          i_to_f = i.to_f
          i_prev = i - 1
          if k == 1
            entry_i = @entries_padded.at(i)
            distance_k[i] = distance_k.at(i_prev) + (i_prev / i_to_f * ((entry_i - mean_x1)**2))
            mean_x1 = ((i_prev * mean_x1) + entry_i) / i_to_f
            backtrack1[i] = 1
          else
            d = 0.0 # the sum of squared distances from x_j ,. . ., x_i to their mean
            mean_xj = 0.0

            i.downto(k) do |j|
              ij_diff = i - j
              ij_diff_offset = ij_diff.to_f + 1
              entry_j = @entries_padded.at(j)
              d += ij_diff / ij_diff_offset * ((entry_j - mean_xj)**2)
              mean_xj = (entry_j + (ij_diff * mean_xj)) / ij_diff_offset
              distance_k_prev_j_prev = distance_k_prev.at(j - 1)

              if j == i
                distance_k[i] = d
                backtrack_k[i] = j
                distance_k[i] += distance_k_prev_j_prev unless j == 1
              elsif j == 1 && d <= distance_k[i]
                distance_k[i] = d
                backtrack_k[i] = j
              elsif d + distance_k_prev_j_prev < distance_k[i]
                distance_k[i] = d + distance_k_prev_j_prev
                backtrack_k[i] = j
              end
            end
          end
        end
      end
    end

    def kopt
      @kopt ||=
        if kmin == kmax
          kmin
        else
          method = :normal # :uniform
          _kopt = kmin

          offset = 1
          n = @entry_count
          n_to_f = n.to_f
          n_to_f_log = Math.log(n_to_f)
          entry_one = entries[0]
          entry_n = entries[-1]

          max_bic = 0.0

          kmin.upto(kmax) do |k|
            cluster_sizes = []

            backtrack_from(k) { |cluster, left, right| cluster_sizes[cluster] = right - left + 1 }

            index_left = offset

            likelihood, index_right, bin_left, bin_right = 0, 0, 0, 0
            k.times do |i|
              cluster_size = cluster_sizes[i + offset]
              index_right = index_left + cluster_size - 1
              index_left_val = @entries_padded[index_left]
              index_right_val = @entries_padded[index_right]

              if index_left_val < index_right_val
                bin_left = index_left_val
                bin_right = index_right_val
              elsif index_left_val == index_right_val
                bin_left = index_left == offset ? entry_one : (@entries_padded[index_left - 1] + index_left_val) / 2
                bin_right = index_right < n ? (index_right_val + @entries_padded[index_right+1]) / 2 : entry_n
              else
                raise RuntimeError.new("Value at left index should not be > value at right index")
              end

              bin_width = bin_right - bin_left
              if method == :uniform
                likelihood += cluster_size * Math.log(cluster_size / bin_width / n)
              else
                mean, variance = 0.0, 0.0

                index_left.upto(index_right) do |j|
                  mean += @entries_padded[j]
                  variance += @entries_padded[j] ** 2
                end
                mean /= cluster_size
                variance = (variance - cluster_size * mean ** 2) / (cluster_size - 1) if cluster_size > 1

                if variance > 0
                  index_left.upto(index_right) do |j|
                    likelihood -= (@entries_padded[j] - mean) ** 2 / (2.0 * variance)
                  end
                  likelihood -= cluster_size * (Math.log(PI_DOUBLE * variance) / 2)
                else
                  likelihood += cluster_size * Math.log(1.0 / bin_width / n)
                end
              end

              index_left = index_right + 1
            end

            bic = 2 * likelihood - (3 * k - 1) * n_to_f_log

            if k == kmin
              max_bic = bic
              _kopt = kmin
            elsif bic > max_bic
              max_bic = bic
              _kopt = k
            end
          end

          _kopt
        end
    end

    def backtrack_from(imax)
      right = backtrack[0].size - 1

      imax.downto(1) do |k|
        left = backtrack[k][right]

        yield k, left, right

        right = left - 1 if k > 1
      end
    end
  end
end
