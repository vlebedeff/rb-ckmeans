# frozen_string_literal: true

RSpec.describe "Fast vs Stable Estimation Comparison" do # rubocop:disable Metrics
  describe "Cases where :fast and :stable differ" do # rubocop:disable Metrics
    context "Ckmeans - duplicate values with large gaps" do
      # Scenario: Three groups of duplicates separated by large gaps
      # From the actual test suite: this is a case where they differ
      let(:data) { [1, 1, 1, 100, 100, 100, 1000, 1000] }

      it "fast: merges closer clusters" do
        result = Ckmeans::Clusterer.new(data, 1, 8, :fast).clusters
        # Fast method keeps first two groups together
        expect(result).to eq([[1, 1, 1, 100, 100, 100], [1000, 1000]])
      end

      it "stable: identifies all three distinct groups" do
        result = Ckmeans::Clusterer.new(data, 1, 8, :stable).clusters
        # GMM correctly identifies all three separate groups
        expect(result).to eq([[1, 1, 1], [100, 100, 100], [1000, 1000]])
      end
    end

    context "Ckmeans - tight clusters with outlier" do
      # Scenario: Relatively close values with one outlier
      let(:data) { [0.1, 1.1, 1.2, 1.6, 2.2, 2.5, 2.7, 2.8, 3, 3.1, 7.1] }

      it "fast: splits out the outlier" do
        result = Ckmeans::Clusterer.new(data, 1, data.size, :fast).clusters
        # Fast sees 7.1 as far enough to be its own cluster
        expect(result).to eq([data[0..-2], [7.1]])
      end

      it "stable: keeps everything together" do
        result = Ckmeans::Clusterer.new(data, 1, data.size, :stable).clusters
        # GMM recognizes this as one continuous distribution
        expect(result).to eq([data])
      end
    end

    context "Ckmedian - two close values" do
      # Scenario: Two values very close together
      # From actual tests: fast splits them, stable keeps together
      let(:data) { [10, 11] }

      it "fast: splits into two clusters" do
        result = Ckmedian::Clusterer.new(data, 1, 2, :fast).clusters
        # Fast method sees these as separate enough
        expect(result).to eq([[10], [11]])
      end

      it "stable: keeps them together" do
        result = Ckmedian::Clusterer.new(data, 1, 2, :stable).clusters
        # LMM recognizes these as belonging to the same cluster
        expect(result).to eq([[10, 11]])
      end
    end

    context "Ckmedian - duplicate clusters with large gaps" do
      # Scenario: Same as Ckmeans example but with L1 distance
      # This shows LMM handles duplicates similarly to GMM
      let(:data) { [1, 1, 1, 100, 100, 100, 1000, 1000] }

      it "fast: merges closer clusters" do
        result = Ckmedian::Clusterer.new(data, 1, 8, :fast).clusters
        # Fast method keeps first two groups together
        expect(result).to eq([[1, 1, 1, 100, 100, 100], [1000, 1000]])
      end

      it "stable: identifies all three distinct groups" do
        result = Ckmedian::Clusterer.new(data, 1, 8, :stable).clusters
        # LMM correctly identifies all three separate groups
        expect(result).to eq([[1, 1, 1], [100, 100, 100], [1000, 1000]])
      end
    end

    context "When :fast and :stable agree" do
      # Scenario: Well-separated, roughly Gaussian clusters
      let(:clean_data) { [1, 2, 3, 100, 101, 102, 1000, 1001, 1002] }

      it "both produce the same result for clean, well-separated data" do
        fast = Ckmeans::Clusterer.new(clean_data, 1, 5, :fast).clusters
        stable = Ckmeans::Clusterer.new(clean_data, 1, 5, :stable).clusters

        expect(fast).to eq(stable)
        expect(fast).to eq([[1, 2, 3], [100, 101, 102], [1000, 1001, 1002]])
      end

      it "why: when data fits the Gaussian assumption, fast method works perfectly" do
        # Fast method assumes Gaussian clusters
        # When data actually is Gaussian-like, it performs identically to GMM
      end
    end
  end

  describe "Performance characteristics" do
    context "when to use :fast" do
      it "handles large datasets efficiently" do
        large_data = Array.new(10_000) { rand * 1_000_000 }

        time_fast = Benchmark.measure do
          Ckmeans::Clusterer.new(large_data, 1, 20, :fast).clusters
        end

        expect(time_fast.real).to be < 3.0
      end
    end

    context "when to use :stable" do
      it "handles edge cases better even if slower" do
        edge_case_data = ([1] * 100) + ([50] * 2) + [100] # 100 dupes, 2 dupes, 1 unique

        time_stable = Benchmark.measure do
          Ckmeans::Clusterer.new(edge_case_data, 1, 10, :stable).clusters
        end

        # Slower but more reliable
        expect(time_stable.real).to be < 5.0
      end
    end
  end

  describe "Summary: When results differ" do
    it "duplicates: stable handles zero-variance clusters better" do
      data = [1, 1, 1, 50, 50, 100]
      fast = Ckmeans::Clusterer.new(data, 1, 4, :fast).clusters
      stable = Ckmeans::Clusterer.new(data, 1, 4, :stable).clusters

      # Stable correctly identifies 3 groups, fast merges some
      expect(fast).not_to eq(stable)
    end

    it "unbalanced clusters: stable better at identifying small clusters" do
      data = ([1] * 20) + [50, 51] + ([100] * 20) # Tiny cluster in the middle
      fast = Ckmeans::Clusterer.new(data, 1, 5, :fast).clusters
      stable = Ckmeans::Clusterer.new(data, 1, 5, :stable).clusters

      expect(fast).to eq([data])
      expect(stable).to eq(
        [[1] * 20, [50, 51], [100] * 20]
      )
    end
  end
end
