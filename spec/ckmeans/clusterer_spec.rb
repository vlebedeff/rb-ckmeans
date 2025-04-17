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
      kfixed = described_class.new(x, 1)
      ksensitive = described_class.new(x, 1, 1, :sensitive)
      expect(kfixed.clusters).to eq([[1]])
      expect(ksensitive.clusters).to eq([[1]])
    end

    specify "two near values" do
      x = [10, 11]
      kfixed = described_class.new(x, 1, 2)
      ksensitive = described_class.new(x, 1, 2, :sensitive)
      expect(kfixed.clusters).to eq([[10], [11]])
      expect(ksensitive.clusters).to eq([[10], [11]])
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

      it "splits sequences with more data points" do
        expect(described_class.new(x + [100_000], 1, 4).clusters).to eq([[100, 100, 100], [99_999, 100_000]])
        expect(described_class.new(x + [100_000], 1, 4, :sensitive).clusters).to(
          eq([[100, 100, 100], [99_999, 100_000]])
        )
      end
    end

    context "uniform positive sequence" do
      let(:x) { [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80] }

      specify do
        expect(described_class.new(x, 1, 15).clusters).to eq([x])
      end

      specify do
        expect(described_class.new(x.shuffle, 1, 15).clusters).to eq([x])
      end

      specify do
        expect(described_class.new(x, 2, 15).clusters).to eq([x[0..7], x[8..15]])
      end

      specify do
        expect(described_class.new(x, 3, 15).clusters).to eq([x[0..5], x[6..10], x[11..15]])
      end

      specify do
        expect(described_class.new(x, 6, 15).clusters).to(
          eq([x[0..2], x[3..5], x[6..8], x[9..11], x[12..13], x[14..15]])
        )
      end

      specify do
        expect(described_class.new(x, 1, 15, :sensitive).clusters).to eq([x[0..3], x[4..7], x[8..11], x[12..15]])
      end

      specify do
        expect(described_class.new(x.shuffle, 1, 15, :sensitive).clusters).to(
          eq([x[0..3], x[4..7], x[8..11], x[12..15]])
        )
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

    it "processes 1000 elements into optimimal cluster count within 2.5s" do
      entries = Array.new(1000) { rand * 10_000.0 }
      clusters = nil
      # profiler = RubyProf::Profile.new
      # profiler.start
      bm = Benchmark.measure { clusters = described_class.new(entries, 1, 1000, :sensitive).clusters }
      # puts bm.total
      # profiling = profiler.stop
      # printer = RubyProf::FlatPrinter.new(profiling)
      # printer.print(STDOUT)
      expect(bm.total).to be < 2.5
    end
  end

  describe "#xsorted_cluster_index" do
    xspecify do
      clusterer = described_class.new([], 0, 0)
      expect(clusterer.xsorted_cluster_index).to be_nil
    end

    specify do
      clusterer = described_class.new([100, 100, 100, 99_999, 100_000], 1, 5)
      expect(clusterer.xsorted_cluster_index).to be_nil
    end
  end
end
