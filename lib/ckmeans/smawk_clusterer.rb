# frozen_string_literal: true

module Ckmeans
  class SmawkClusterer # rubocop:disable Style/Documentation, Metrics/ClassLength
    attr_reader :xcount, :xsorted, :kmin, :kmax, :smat, :jmat

    PI_DOUBLE = Math::PI * 2

    def initialize(entries, kmin, kmax = kmin)
      @xcount = entries.size

      raise ArgumentError, "Minimum cluster count is bigger than element count" if kmin > @xcount
      raise ArgumentError, "Maximum cluster count is bigger than element count" if kmax > @xcount

      @kmin = kmin
      @unique_xcount = entries.uniq.size
      @kmax = [@unique_xcount, kmax].min
      @xsorted = entries.sort
    end

    def clusters # rubocop:disable Metrics/MethodLength
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
          jseq = Array.new
          shift = xsorted[n / 2]
          xsum[0] = xsorted[0] - shift
          xsumsq[0] = xsum[0]**2
          1.upto(n - 1) do |i|
            xsum[i] = xsum[i - 1] + xsorted[i] - shift
            xsumsq[i] = xsumsq[i - 1] + (xsorted[i] - shift) * (xsorted[i] - shift)
            smat[0][i] = dissim(0, i, xsum, xsumsq)
            jmat[0][i] = 0
          end

          kappa_dec = kappa - 1
          1.upto(kappa_dec) do |q|
            imin = q < kappa_dec ? [1, q].max : n - 1
            fill_row_q_linear(imin, n - 1, q, smat, jmat, xsum, xsumsq)
          end

          kopt = select_levels_3_4_12(jmat)

          results = []
          backtrack(jmat, kopt) do |q, left, right|
            results[q] = xsorted[left..right]
          end
          results
        end
    end

    private

    def range_of_variance(x)
      dposmin = x[-1] - x[0]
      dposmax = 0.0

      (1...x.size).each do |n|
        d = x[n] - x[n-1]
        dposmin = d if d > 0 && dposmin > d
        dposmax = d if d > dposmax
      end

      variance_min = dposmin * dposmin / 3.0
      variance_max = dposmax * dposmax

      [variance_min, variance_max]
    end

    def select_levels_3_4_12(jmat)
      method = "normal" # "uniform" or "normal"
      kopt = kmin
      base = 0
      n = xsorted.size - base
      max_bic = 0.0
      bic_values = {}

      kmin.upto(kmax) do |k|
        sizes = Array.new(k + base)
        backtrack(jmat, k) { |q, left, right| sizes[q + base] = right - left + 1 }
        index_left = base
        index_right = nil
        loglikelihood = 0.0
        bin_left = nil
        bin_right = nil

        k.times do |kb|
          num_points_in_bin = sizes[kb + base]
          index_right = index_left + num_points_in_bin - 1

          if xsorted[index_left] < xsorted[index_right]
            bin_left = xsorted[index_left]
            bin_right = xsorted[index_right]
          elsif xsorted[index_left] == xsorted[index_right]
            bin_left = (index_left == base) ?
                       xsorted[base] :
                       (xsorted[index_left-1] + xsorted[index_left]) / 2.0
            bin_right = (index_right < n-1+base) ?
                        (xsorted[index_right] + xsorted[index_right+1]) / 2.0 :
                        xsorted[n-1+base]
          else
            raise "ERROR: binLeft > binRight"
          end

          bin_width = bin_right - bin_left

          if method == "uniform"
            density = num_points_in_bin / bin_width / n
            loglikelihood += num_points_in_bin * Math.log(density)
          else # normal
            mean, variance = shifted_data_variance(index_left, index_right)

            if variance > 0
              (index_left..index_right).each do |i|
                loglikelihood += -(xsorted[i] - mean) * (xsorted[i] - mean) / (2.0 * variance)
              end
              loglikelihood += num_points_in_bin *
                               (Math.log(num_points_in_bin / n.to_f) -
                               0.5 * Math.log(2.0 * Math::PI * variance))
            else
              loglikelihood += num_points_in_bin * Math.log(1.0 / bin_width / n)
            end
          end

          index_left = index_right + 1
        end

        bic = if method == "uniform"
                2.0 * loglikelihood - (3 * k - 1) * Math.log(n)
              elsif method == "normal"
                2.0 * loglikelihood - (3 * k - 1) * Math.log(n.to_f)
              end

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
      sum, sumsq, mean, variance = 0.0, 0.0, 0.0, 0.0
      n = iright - ileft + 1

      if iright >= ileft
        median = xsorted[(ileft + iright) / 2]

        ileft.upto(iright) do |i|
          sumi = xsorted[i] - median
          sum += sumi
          sumsq += sumi**2
        end

        mean = sum / n + median
        variance = (sumsq - sum * sum / n) / (n - 1) if n > 1
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
      sji = 0.0

      if j >= i
        sji = 0.0
      elsif j > 0
        muji = (xsum[i] - xsum[j - 1]) / (i - j + 1)
        sji = xsumsq[i] - xsumsq[j - 1] - (i - j + 1) * muji * muji
      else
        sji = xsumsq[i] - xsum[i] * xsum[i] / (i + 1)
      end

      [0, sji].max
    end


    def fill_row_q_linear(imin, imax, q, smat, jmat, xsum, xsumsq)
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
        imax_odd = imin_odd + (imax - imin_odd) / istepx2 * istepx2
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

        ((rmin+1)...js.size).each do |r|
          jabs = js[r]

          next if jabs < jmat[q - 1][i]
          break if jabs > i

          sj = smat[q - 1][jabs - 1] + dissim(jabs, i, xsum, xsumsq)

          if sj <= smat[q][i]
            smat[q][i] = sj
            jmat[q][i] = js[r]
            rmin_prev = r
          end
        end
      end
    end

    def js_reduced(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
      n = (imax - imin) / istep + 1
      js_red = js.dup

      return js_red if n >= js.size

      left, right = -1, 0
      m = js_red.size

      while m > n
        p = left + 1
        i = imin + p * istep
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

      ((left+1)...m).each do |r|
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
        while js[r] < jl
          r += 1
        end

        smat[q][i] = smat[q - 1][js[r] - 1] + dissim(js[r], i, xsum, xsumsq)
        jmat[q][i] = js[r]
        jh         = (((i + istep) <= imax) ? jmat[q][i + istep] : js[n - 1]).to_i
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
