# frozen_string_literal: true

RSpec.describe Ckmedian::Clusterer do # rubocop:disable Metrics/BlockLength
  describe "#clusters" do # rubocop:disable Metrics/BlockLength
    specify do
      x = [-1, 2, 4, 5, 6]
      clusters = described_class.new(x, 3).clusters

      expect(clusters).to eq([[-1], [2], [4, 5, 6]])
    end

    specify do
      x = [-0.9, 1, 1.1, 1.9, 2, 2.05]
      clusters = described_class.new(x, 1, 6).clusters

      expect(clusters).to eq([[-0.9], [1, 1.1], [1.9, 2, 2.05]])
    end

    context "relatively close values" do
      let(:x) { [0.1, 1.1, 1.2, 1.6, 2.2, 2.5, 2.7, 2.8, 3, 3.1, 7.1] }

      example "fast" do
        expect(described_class.new(x, 1, 8).clusters).to eq([x])
      end

      example "lmm" do
        expect(described_class.new(x, 1, 8, :lmm).clusters).to eq([x])
      end
    end

    describe "two near values" do
      let(:x) { [10, 11] }

      specify "fast" do
        clusters = described_class.new(x, 1, 2).clusters
        expect(clusters).to eq([[10], [11]])
      end

      specify "stable (using :lmm)" do
        clusters = described_class.new(x, 1, 2, :lmm).clusters
        expect(clusters).to eq([[10, 11]])
      end

      specify "stable (using :stable alias)" do
        clusters = described_class.new(x, 1, 2, :stable).clusters
        expect(clusters).to eq([[10, 11]])
      end
    end

    specify "multiple dupes of one unique value" do
      x = [1, 1, 1, 1]

      fast = described_class.new(x, 1, 4).clusters
      lmm = described_class.new(x, 1, 4, :lmm).clusters

      expect(fast).to eq(lmm).and eq([x])
    end

    example "large gaps and dupes" do
      x = [1, 1, 1, 100, 100, 100, 1000, 1000]
      kmin = 1
      kmax = 8
      fast = described_class.new(x, kmin, kmax).clusters
      lmm = described_class.new(x, kmin, kmax, :lmm).clusters

      expect(fast).to eq([[1, 1, 1, 100, 100, 100], [1000, 1000]])
      expect(lmm).to eq([[1, 1, 1], [100, 100, 100], [1000, 1000]])
    end

    example "large gaps distinct" do
      x = [1, 2, 3, 100, 101, 102, 1000, 1001]

      fast = described_class.new(x, 1, x.size).clusters
      lmm = described_class.new(x, 1, x.size, :lmm).clusters

      expect(fast).to eq(lmm).and eq(
        [[1, 2, 3], [100, 101, 102], [1000, 1001]]
      )
    end

    context "bimodal distributions" do
      let(:x) { [1, 1.1, 1.2, 2, 2.1, 2.2, 15, 15.1, 15.2] }
      let(:fast) { described_class.new(x, 1, x.size).clusters }
      let(:lmm) { described_class.new(x, 1, x.size, :lmm).clusters }

      specify do
        expect(fast).to eq(lmm).and eq(
          [[1, 1.1, 1.2], [2, 2.1, 2.2], [15, 15.1, 15.2]]
        )
      end
    end

    context "varying density" do
      let(:x) { [0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5] }
      let(:fast) { described_class.new(x, 1, x.size).clusters }
      let(:lmm) { described_class.new(x, 1, x.size, :lmm).clusters }

      specify do
        expect(fast).to eq(lmm).and eq(
          [[0.1, 0.2, 0.3, 1.0, 2.0, 3.0], [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]]
        )
      end
    end

    describe "performance" do
      # rubocop:disable Lint/ConstantDefinitionInBlock -- in order to use the values in example descriptions
      N_FAST_L1 = 5_000
      N_LMM = N_FAST_L1 / 10
      # rubocop:enable Lint/ConstantDefinitionInBlock

      let(:x) { Array.new(N_FAST_L1) { rand * 1_000_000 } }
      let(:fast) { described_class.new(x, 1, x.size).clusters }
      let(:lmm) { described_class.new(x.first(N_LMM), 1, N_LMM, :lmm).clusters }

      specify "`:fast` mode handles #{N_FAST_L1} elements and K estimation within 3s" do
        bm = Benchmark.measure { fast }
        expect(bm.total).to be < 3
      end

      specify "`:lmm` mode handles #{N_LMM} elements and K estimation within 3s" do
        bm = Benchmark.measure { lmm }
        expect(bm.total).to be < 3
      end
    end
  end
end
