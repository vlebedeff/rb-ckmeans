# frozen_string_literal: true

RSpec.describe Ckmeans::Clusterer do # rubocop:disable Metrics/BlockLength
  describe "#new" do
    specify do
      expect { described_class.new([], 0, 0) }.to_not raise_error
      expect { described_class.new([], 0) }.to_not raise_error
    end

    context "when min cluster count > element count" do
      specify do
        expect { described_class.new([], 1) }.to(
          raise_error(ArgumentError, "Minimum cluster count is bigger than element count")
        )
      end
    end

    context "when min cluster count > element count" do
      specify do
        expect { described_class.new([], 0, 1) }.to(
          raise_error(ArgumentError, "Maximum cluster count is bigger than element count")
        )
      end
    end
  end

  describe "#clusters" do # rubocop:disable Metrics/BlockLength
    specify "one value" do
      x = [1]
      clusters = described_class.new(x, 1).clusters
      expect(clusters).to eq([[1]])
    end

    describe "two near values" do
      let(:x) { [10, 11] }

      specify "fast" do
        clusters = described_class.new(x, 1, 2).clusters
        expect(clusters).to eq([[10], [11]])
      end

      specify "stable" do
        clusters = described_class.new(x, 1, 2, :gmm).clusters
        expect(clusters).to eq([[10, 11]])
      end
    end

    specify "multiple dupes of one unique value" do
      x = [1, 1, 1, 1]

      fast = described_class.new(x, 1, 4).clusters
      stable = described_class.new(x, 1, 4, :gmm).clusters

      expect(fast).to eq(stable).and eq([x])
    end

    context "small set with two unique values" do
      let(:x) { [100, 100, 100, 99_999] }

      it "is split with explicit kmin" do
        expect(described_class.new(x, 2).clusters).to eq([[100, 100, 100], [99_999]])
      end

      it "penalizes extra clusters" do
        fast = described_class.new(x, 1, 4).clusters
        stable = described_class.new(x, 1, 4, :gmm).clusters

        expect(fast).to eq(stable).and eq([x])
      end

      it "splits sequences with more data points" do
        expect(
          described_class.new(x + [100_000], 1, 4).clusters
        ).to eq([[100, 100, 100], [99_999, 100_000]])
      end
    end

    context "uniform positive sequence" do
      let(:x) { [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80] }

      it "unconstrained K produces one cluster" do
        expect(described_class.new(x, 1, 15).clusters).to eq([x])
      end

      it "unconstrained K produces one cluster for unsorted collection" do
        expect(described_class.new(x.shuffle, 1, 15).clusters).to eq([x])
      end

      specify "K >= 2 produces two roughly equally sized clusters" do
        expect(described_class.new(x, 2, 15).clusters).to eq([x[0..7], x[8..15]])
      end

      specify "K >= 3 produces three roughly equally sized clusters" do
        expect(described_class.new(x, 3, 15).clusters).to eq([x[0..5], x[6..10], x[11..15]])
      end

      specify "K >= 6 produces six roughly equally sized clusters" do
        expect(described_class.new(x, 6, 15).clusters).to(
          eq([x[0..2], x[3..5], x[6..8], x[9..11], x[12..13], x[14..15]])
        )
      end
    end

    example "large gaps and dupes" do
      x = [1, 1, 1, 100, 100, 100, 1000, 1000]
      kmin = 1
      kmax = 8
      fast = described_class.new(x, kmin, kmax).clusters
      stable = described_class.new(x, kmin, kmax, :gmm).clusters

      expect(fast).to eq([[1, 1, 1, 100, 100, 100], [1000, 1000]])
      expect(stable).to eq([[1, 1, 1], [100, 100, 100], [1000, 1000]])
    end

    example "large gaps distinct" do
      x = [1, 2, 3, 100, 101, 102, 1000, 1001]

      fast = described_class.new(x, 1, x.size).clusters
      stable = described_class.new(x, 1, x.size, :gmm).clusters

      expect(fast).to eq(stable).and eq(
        [[1, 2, 3], [100, 101, 102], [1000, 1001]]
      )
    end

    example do
      x = [3.5, 3.6, 3.7, 3.1, 1.1, 0.9, 0.8, 2.2, 1.9, 2.1]

      fast = described_class.new(x, 1, x.size).clusters
      stable = described_class.new(x, 1, x.size, :gmm).clusters

      expect(fast).to eq(stable).and eq(
        [[0.8, 0.9, 1.1], [1.9, 2.1, 2.2], [3.1, 3.5, 3.6, 3.7]]
      )
    end

    specify do
      expect(
        described_class.new([0.9, 1, 1.1, 1.9, 2, 2.1], 1, 6).clusters
      ).to eq([[0.9, 1, 1.1], [1.9, 2, 2.1]])
    end

    context "relatively close values" do
      let(:x) { [0.1, 1.1, 1.2, 1.6, 2.2, 2.5, 2.7, 2.8, 3, 3.1, 7.1] }

      example "fast" do
        expect(described_class.new(x, 1, x.size).clusters).to eq([x[0..-2], [7.1]])
      end

      example "stable" do
        expect(described_class.new(x, 1, x.size, :gmm).clusters).to eq([x])
      end
    end

    context "bimodal distributions" do
      let(:x) { [1, 1.1, 1.2, 2, 2.1, 2.2, 15, 15.1, 15.2] }
      let(:fast) { described_class.new(x, 1, x.size).clusters }
      let(:stable) { described_class.new(x, 1, x.size, :gmm).clusters }

      specify do
        expect(fast).to eq(stable).and eq(
          [[1, 1.1, 1.2], [2, 2.1, 2.2], [15, 15.1, 15.2]]
        )
      end
    end

    context "varying density" do
      let(:x) { [0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5] }
      let(:fast) { described_class.new(x, 1, x.size).clusters }
      let(:stable) { described_class.new(x, 1, x.size, :gmm).clusters }

      specify do
        expect(fast).to eq(stable).and eq(
          [[0.1, 0.2, 0.3, 1.0, 2.0, 3.0], [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]]
        )
      end
    end

    context "yet another set" do
      let(:x) { [1, 1.1, 1.2, 5, 10, 20, 30, 40, 50] }
      let(:fast) { described_class.new(x, 1, x.size).clusters }
      let(:stable) { described_class.new(x, 1, x.size, :gmm).clusters }

      specify do
        expect(fast).to eq(stable).and eq([x])
      end
    end

    context "cosine sequence" do
      let(:x) { (-10..10).map { |i| Math.cos(i) } }
      let(:fast) { described_class.new(x, 1, x.size).clusters }
      let(:stable) { described_class.new(x, 1, x.size, :gmm).clusters }

      specify do
        expect(fast).to eq(stable)
        expect(fast.size).to eq(2)
        expect(fast.first).to all(be_negative)
        expect(fast.last).to all(be_positive)
      end
    end

    specify do
      expect(
        described_class.new([-1, 2, -1, 2, 4, 5, 6, -1, 2, -1], 3).clusters
      ).to eq([[-1, -1, -1, -1], [2, 2, 2], [4, 5, 6]])
    end

    specify do
      expect(described_class.new([3, 2, -5.4, 0.1], 4).clusters).to eq([[-5.4], [0.1], [2], [3]])
    end

    specify do
      expect(described_class.new(Array.new(10) { |i| i + 1 }, 2).clusters).to eq([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    end

    specify do
      expect(described_class.new([-3, 2.2, -6, 7, 9, 11, -6.3, 75, 82.6, 32.3, -9.5, 62.5, 7, 95.2], 8).clusters).to(
        eq([[-9.5, -6.3, -6], [-3, 2.2], [7, 7, 9, 11], [32.3], [62.5], [75], [82.6], [95.2]])
      )
    end

    describe "performance" do
      # rubocop:disable Lint/ConstantDefinitionInBlock -- in order to use the values in example descriptions
      N_FAST = 5_000
      N_STABLE = N_FAST / 10
      # rubocop:enable Lint/ConstantDefinitionInBlock

      let(:x) { Array.new(N_FAST) { rand * 1_000_000 } }
      let(:fast) { described_class.new(x, 1, x.size).clusters }
      let(:stable) { described_class.new(x.first(N_STABLE), 1, N_STABLE, :gmm).clusters }

      specify "`:fast` mode handles #{N_FAST} elements and K estimation within 3s" do
        bm = Benchmark.measure { fast }
        # puts bm.total
        expect(bm.total).to be < 3
      end

      specify "`:stable` mode handles #{N_STABLE} elements and K estimation within 3s" do
        bm = Benchmark.measure { stable }
        # puts bm.total
        expect(bm.total).to be < 3
      end
    end
  end
end
