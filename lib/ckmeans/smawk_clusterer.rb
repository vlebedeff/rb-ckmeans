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

          # fill matrix start
          kappa = kmax
          n = xcount
          xsum = Array.new(n)
          xsumsq = Array.new(n)
          jseq = Array.new
          shift = xsorted[n / 2]
          xsum[0] = xsorted[0] - shift
          xsumsq[0] = xsum[0]**2
          # smat[0][0] = 0.0
          # jmat[0][0] = 0
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
          # fill matrix end
          kopt = select_levels(jmat)

          results = []
          backtrack(jmat, kopt) do |q, left, right|
            results[q] = xsorted[left..right]
          end
          results
        end
    end

    private

    def select_levels(jmat)
      return [kmin, kmax].min if kmin > kmax || xcount < 2

      kopt = kmin
      bicmax = 0.0

      lambda_ = Array.new(kmax)
      mu = Array.new(kmax)
      sigma2 = Array.new(kmax)
      coeff = Array.new(kmax)

      kmin.upto(kmax) do |kappa|
        sizes = Array.new(kappa)

        backtrack(jmat, kappa) { |q, left, right| sizes[q] = right - left + 1 }

        ileft = 0
        iright = nil

        kappa.times do |k|
          lambda_[k] = sizes[k] / xcount.to_f
          iright = ileft + sizes[k] - 1

          mu[k], sigma2[k] = shifted_data_variance(ileft, iright)

          if sigma2[k] == 0 || sizes[k] == 1
            dmin =
              if (ileft > 0) && (iright < (xcount - 1))
                [xsorted[ileft] - xsorted[ileft - 1], xsorted[iright + 1] - xsorted[iright]].min
              elsif ileft > 0
                xsorted[ileft] - xsorted[ileft - 1]
              else
                xsorted[iright + 1] - xsorted[iright]
              end

            sigma2[k] = dmin * dmin / 4.0 / 9.0 if sigma2[k] == 0
            sigma2[k] = dmin * dmin if sizes[k] == 1
          end

          coeff[k] = lambda_[k] / Math.sqrt(2.0 * Math::PI * sigma2[k])

          ileft = iright + 1
        end

        loglikelihood = 0

        xcount.times do |i|
          likelihood = 0.0

          kappa.times do |k|
            likelihood += coeff[k] * Math.exp( - (xsorted[i] - mu[k]) ** 2 / (2.0 * sigma2[k]))
          end

          loglikelihood += Math.log(likelihood)
        end

        bic = 2 * loglikelihood - (3 * kappa - 1) * Math.log(xcount.to_f)

        if kappa == kmin
          bicmax = bic
          kopt = kmin
        else
          bicmax, kopt = bic, kappa if bic > bicmax
        end
      end

      kopt
    end

    def shifted_data_variance(ileft, iright)
      sum, sumsq, mean, variance = 0.0, 0.0, 0.0, 0.0
      n = iright - ileft + 1

      if iright > ileft
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
      js = (q...(imax+1)).to_a
      smawk(imin, imax, 1, q, js, smat, jmat, xsum, xsumsq)
    end

    def smawk(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
      if (imax - imin) <= (0 * istep)
        find_min_from_candidates(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
      else
        js_odd = reduce_in_place(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
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

    # TODO: Rename to `js_reduced`
    def reduce_in_place(imin, imax, istep, q, js, smat, jmat, xsum, xsumsq)
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
        if jplus1 == nil
          require 'pry'
          binding.pry
        end
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
          if r >= n
            require 'pry'
            binding.pry
          end
        end

        smat[q][i] = smat[q - 1][js[r] - 1] + dissim(js[r], i, xsum, xsumsq)
        jmat[q][i] = js[r]
        jh         = ((i + istep <= imax) ? jmat[q][i + istep] : js[n - 1]).to_i
        jmax       = [jh, i].min.to_i
        sjimin     = dissim(jmax, i, xsum, xsumsq)

        r += 1
        while r < n && js[r] <= jmax
          jabs = js[r]

          break if jabs > i
          next if jabs < jmat[q - 1][i]

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
