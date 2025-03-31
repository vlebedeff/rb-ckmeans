# frozen_string_literal: true

RSpec.describe Ckmeans::SmawkClusterer do
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

  describe "#clusters" do
    specify do
      x = [1]
      kfixed = described_class.new(x, 1)
      ksensitive = described_class.new(x, 1, 1, :sensitive)
      expect(kfixed.clusters).to eq([[1]])
      expect(ksensitive.clusters).to eq([[1]])
    end

    specify do
      x = [1, 1, 1, 1]
      kflexible = described_class.new(x, 1, 4)
      ksensitive = described_class.new(x, 1, 4, :sensitive)
      expect(kflexible.clusters).to eq([[1, 1, 1, 1]])
      expect(ksensitive.clusters).to eq([[1, 1, 1, 1]])
    end

    context "small set with two unique values" do
      let(:x) { [100, 100, 100, 99_999] }

      it "is split with explicit kmin" do
        expect(described_class.new(x, 2).clusters).to eq([[100, 100, 100], [99_999]])
      end

      it "penalizes extra clusters with regular sensitivity" do
        expect(described_class.new(x, 1, 4).clusters).to eq([[100, 100, 100, 99_999]])
      end

      it "penalizes extra clusters with high sensitivity" do
        expect(described_class.new(x, 1, 4, :sensitive).clusters).to eq([[100, 100, 100, 99_999]])
      end
    end

    specify do
      expect(described_class.new([1, 1, 1, 100, 100, 100, 1000, 1000], 1,
                                 8).clusters).to eq([[1, 1, 1, 100, 100, 100], [1000, 1000]])
    end

    specify do
      expect(described_class.new([1, 1, 1, 100, 100, 100, 1000, 1000], 1, 3,
                                 :sensitive).clusters).to eq([[1, 1, 1, 100, 100, 100], [1000, 1000]])
    end

    specify do
      expect(described_class.new([3.5, 3.6, 3.7, 3.1, 1.1, 0.9, 0.8, 2.2, 1.9, 2.1], 1,
                                 5).clusters).to eq([[0.8, 0.9, 1.1], [1.9, 2.1, 2.2], [3.1, 3.5, 3.6, 3.7]])
    end

    specify do
      expect(described_class.new([0.9, 1, 1.1, 1.9, 2, 2.1], 1, 6).clusters).to eq([[0.9, 1, 1.1], [1.9, 2, 2.1]])
    end

    context "relatively close values" do
      let(:x) { [0.1, 1.1, 1.2, 1.6, 2.2, 2.5, 2.7, 2.8, 3, 3.1, 7.1] }

      specify "regular sensitivity" do
        expect(described_class.new(x, 1, 8).clusters).to eq([[0.1, 1.1, 1.2, 1.6, 2.2, 2.5, 2.7, 2.8, 3, 3.1], [7.1]])
      end

      specify "high sensitivity" do
        expect(described_class.new(x, 1, 8, :sensitive).clusters).to(
          eq([[0.1, 1.1, 1.2, 1.6], [2.2, 2.5, 2.7, 2.8, 3, 3.1], [7.1]])
        )
      end
    end

    context "cosine sequence" do
      let(:x) { (-10..10).map { |i| Math.cos(i) } }

      specify "regular sensitivity" do
        clusters = described_class.new(x, 1, x.size).clusters
        expect(clusters.size).to eq(2)
        expect(clusters.first).to all(be_negative)
        expect(clusters.last).to all(be_positive)
      end

      specify "high sensitivity" do
        clusters = described_class.new(x, 1, x.size, :sensitive).clusters
        expect(clusters.size).to eq(4)
        expect(clusters.first).to all(be_negative)
        expect(clusters[1]).to all(be_negative)
        expect(clusters[2]).to all(be_positive)
        expect(clusters.last).to all(be_positive)
      end
    end

    specify do
      expect(described_class.new([-1, 2, -1, 2, 4, 5, 6, -1, 2, -1],
                                 3).clusters).to eq([[-1, -1, -1, -1], [2, 2, 2], [4, 5, 6]])
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

    it "processes 1000 elements into optimimal cluster count within 3s" do
      entries = Array.new(1000) { rand * 10_000.0 }
      clusters = nil
      bm = Benchmark.measure { clusters = described_class.new(entries, 1, 1000, :sensitive).clusters }
      expect(bm.total).to be < 3
    end
  end
end
