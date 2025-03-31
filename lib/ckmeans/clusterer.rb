# frozen_string_literal: true

module Ckmeans
  class Clusterer # rubocop:disable Style/Documentation, Metrics/ClassLength
    attr_reader :xcount, :xsorted, :kmin, :kmax, :smat, :jmat, :kestimate

    PI_DOUBLE = Math::PI * 2

    def initialize(entries, kmin, kmax = kmin, kestimate = :regular)
      @xcount = entries.size

      raise ArgumentError, "Minimum cluster count is bigger than element count" if kmin > @xcount
      raise ArgumentError, "Maximum cluster count is bigger than element count" if kmax > @xcount

      @kmin = kmin
      @unique_xcount = entries.uniq.size
      @kmax = [@unique_xcount, kmax].min
      @xsorted = entries.sort
      @kestimate = kestimate
    end

    def clusters
      @clusters ||=
        if @unique_xcount <= 1
          [xsorted]
        else
          @smat = Array.new(kmax) { Array.new(xcount) { 0.0 } }
          @jmat = Array.new(kmax) { Array.new(xcount) { 0 } }

          kappa = kmax
          n = xcount
          xsum = Array.new(n)
          xsumsq = Array.new(n)
          shift = xsorted[n / 2]
          xsum[0] = xsorted[0] - shift
          xsumsq[0] = xsum[0]**2
          1.upto(n - 1) do |i|
            xsum[i] = xsum[i - 1] + xsorted[i] - shift
            xsumsq[i] = xsumsq[i - 1] + ((xsorted[i] - shift) * (xsorted[i] - shift))
            smat[0][i] = dissim(0, i, xsum, xsumsq)
            jmat[0][i] = 0
          end

          kappa_dec = kappa - 1
          1.upto(kappa_dec) do |q|
            imin = q < kappa_dec ? [1, q].max : n - 1
            fill_row(q, imin, n - 1, smat, jmat, xsum, xsumsq)
          end

          kopt = koptimal(jmat)

          results = []
          backtrack(jmat, kopt) do |q, left, right|
            results[q] = xsorted[left..right]
          end
          results
        end
    end

    private

    def koptimal(jmat)
      kopt = kmin
      n = xcount
      max_bic = 0.0

      # Deviation from BIC formula to favor smaller clusters
      adjustment = kestimate == :sensitive ? 0.0 : 1.0

      kmin.upto(kmax) do |k|
        sizes = Array.new(k)
        backtrack(jmat, k) { |q, left, right| sizes[q] = right - left + 1 }
        index_left = 0
        index_right = nil
        loglikelihood = 0.0
        bin_left = nil
        bin_right = nil

        k.times do |kb|
          num_points_in_bin = sizes[kb]
          index_right = index_left + num_points_in_bin - 1

          if xsorted[index_left] < xsorted[index_right]
            bin_left = xsorted[index_left]
            bin_right = xsorted[index_right]
          elsif xsorted[index_left] == xsorted[index_right]
            bin_left = index_left == 0 ? xsorted[0] : (xsorted[index_left - 1] + xsorted[index_left]) / 2.0
            bin_right = index_right < n - 1 ? (xsorted[index_right] + xsorted[index_right + 1]) / 2.0 : xsorted[n - 1]
          else
            raise "ERROR: binLeft > binRight"
          end

          bin_width = bin_right - bin_left

          mean, variance = shifted_data_variance(index_left, index_right)

          if variance > 0
            (index_left..index_right).each do |i|
              loglikelihood += -(xsorted[i] - mean) * (xsorted[i] - mean) / (2.0 * variance)
            end
            loglikelihood +=
              num_points_in_bin *
              ((Math.log(num_points_in_bin / n.to_f) * adjustment) - (0.5 * Math.log(PI_DOUBLE * variance)))
          else
            loglikelihood += num_points_in_bin * Math.log(1.0 / bin_width / n)
          end

          index_left = index_right + 1
        end

        bic = (2.0 * loglikelihood) - (((3 * k) - 1) * Math.log(n.to_f))

        if k == kmin
          max_bic = bic
          kopt = kmin
        elsif bic > max_bic
          max_bic = bic
          kopt = k
        end
      end

      kopt
    end

    def shifted_data_variance(ileft, iright)
      sum = 0.0
      sumsq = 0.0
      mean = 0.0
      variance = 0.0
      n = iright - ileft + 1

      if iright >= ileft
        median = xsorted[(ileft + iright) / 2]

        ileft.upto(iright) do |i|
          sumi = xsorted[i] - median
          sum += sumi
          sumsq += sumi**2
        end

        mean = (sum / n) + median
        variance = (sumsq - (sum * sum / n)) / (n - 1) if n > 1
      end

      [mean, variance]
    end

    def backtrack(jmat, k)
      n = jmat[0].size
      right = n - 1
      left = nil

      (k - 1).downto(0) do |q|
        left = jmat[q][right]

        yield q, left, right

        right = left - 1 if q > 0
      end
    end

    def dissim(j, i, xsum, xsumsq)
      return 0.0 if j >= i

      sji =
        if j > 0
          segment_sum = xsum[i] - xsum[j - 1]
          segment_size = i - j + 1
          xsumsq[i] - xsumsq[j - 1] - (segment_sum * segment_sum / segment_size)
        else
          xsumsq[i] - (xsum[i] * xsum[i] / (i + 1))
        end

      [0, sji].max
    end

    def fill_row(q, imin, imax, smat, jmat, xsum, xsumsq)
      size = imax - q + 1

      js = Array.new(size) { |i| q + i }
      smawk(imin, imax, 1, q, js, smat, jmat, xsum, xsumsq)
    end

    def smawk(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
      if (imax - imin) <= (0 * istep)
        find_min_from_candidates(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
      else
        js_odd = js_reduced(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
        istepx2 = istep * 2
        imin_odd = imin + istep
        imax_odd = imin_odd + ((imax - imin_odd) / istepx2 * istepx2)
        smawk(imin_odd, imax_odd, istepx2, q, js_odd, smat, jmat, xsum, xsumsq)
        fill_even_positions(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
      end
    end

    def find_min_from_candidates(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
      rmin_prev = 0

      (imin..imax).step(istep) do |i|
        rmin = rmin_prev
        smat[q][i] = smat[q - 1][js[rmin] - 1] + dissim(js[rmin], i, xsum, xsumsq)
        jmat[q][i] = js[rmin]

        ((rmin + 1)...js.size).each do |r|
          jabs = js[r]

          next if jabs < jmat[q - 1][i]
          break if jabs > i

          sj = smat[q - 1][jabs - 1] + dissim(jabs, i, xsum, xsumsq)

          next unless sj <= smat[q][i]

          smat[q][i] = sj
          jmat[q][i] = js[r]
          rmin_prev = r
        end
      end
    end

    def js_reduced(imin, imax, istep, q, js, smat, _jmat, xsum, xsumsq)
      n = ((imax - imin) / istep) + 1
      m = js.size

      return js if n >= m

      js_red = js.dup
      left = -1
      right = 0

      while m > n
        p = left + 1
        i = imin + (p * istep)
        j = js_red[right]
        sl = smat[q - 1][j - 1] + dissim(j, i, xsum, xsumsq)
        jplus1 = js_red[right + 1]
        splus1 = smat[q - 1][jplus1 - 1] + dissim(jplus1, i, xsum, xsumsq)

        if (sl < splus1) && (p < n - 1)
          left += 1
          js_red[left] = j
          right += 1
        elsif (sl < splus1) && (p == n - 1)
          right += 1
          js_red[right] = j
          m -= 1
        else
          if p > 0
            js_red[right] = js_red[left]
            left -= 1
          else
            right += 1
          end

          m -= 1
        end
      end

      ((left + 1)...m).each do |r|
        js_red[r] = js_red[right]
        right += 1
      end

      js_red.slice!(m..-1) if js_red.size > m
      js_red
    end

    def fill_even_positions(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
      n = js.size
      istepx2 = istep * 2
      jl = js[0]

      i = imin
      r = 0
      while i <= imax
        r += 1 while js[r] < jl

        smat[q][i] = smat[q - 1][js[r] - 1] + dissim(js[r], i, xsum, xsumsq)
        jmat[q][i] = js[r]
        jh         = ((i + istep) <= imax ? jmat[q][i + istep] : js[n - 1]).to_i
        jmax       = [jh, i].min.to_i
        sjimin     = dissim(jmax, i, xsum, xsumsq)

        r += 1
        while r < n && js[r] <= jmax
          jabs = js[r]

          break if jabs > i

          if jabs < jmat[q - 1][i]
            r += 1
            next
          end

          s  = dissim(jabs, i, xsum, xsumsq)
          sj = smat[q - 1][jabs - 1] + s

          if sj <= smat[q][i]
            smat[q][i] = sj
            jmat[q][i] = js[r]
          elsif smat[q - 1][jabs - 1] + sjimin > smat[q][i]
            break
          end

          r += 1
        end

        r -= 1
        jl = jh

        i += istepx2
      end
    end
  end
end
