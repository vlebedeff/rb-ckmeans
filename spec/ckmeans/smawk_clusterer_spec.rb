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
      expect(described_class.new([1], 1).clusters).to eq([[1]])
    end

    specify do
      expect(described_class.new([1, 1, 1, 1], 2).clusters).to eq([[1, 1, 1, 1]])
    end

    specify do
      expect(described_class.new([100, 100, 100, 99_999], 2).clusters).to eq([[100, 100, 100], [99_999]])
    end

    specify do
      expect(described_class.new([1, 1, 1, 100, 100, 100, 1000, 1000], 1, 2).clusters).to eq([[1, 1, 1, 100, 100, 100], [1000, 1000]])
    end

    specify do
      expect(described_class.new([0.1, 1.1, 1.2, 1.6, 2.2, 2.5, 2.7, 2.8, 3, 3.1, 7.1], 1, 4).clusters).to eq([[0.1], [1.1, 1.2, 1.6], [2.2, 2.5, 2.7, 2.8, 3, 3.1], [7.1]])
    end

    specify do
      expect(described_class.new([-1, 2, -1, 2, 4, 5, 6, -1, 2, -1], 3).clusters).to eq([[-1, -1, -1, -1], [2, 2, 2], [4, 5, 6]])
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

    specify do
      expect(described_class.new([0.9, 1, 1.1, 1.9, 2, 2.1], 1, 6).clusters).to eq([[0.9, 1, 1.1], [1.9, 2, 2.1]])
    end

    it "processes 500 elements into optimimal cluster count within 3s" do
      entries = Array.new(500) { rand * 1000.0 }
      bm = Benchmark.measure { described_class.new(entries, 1, 500).clusters }
      expect(bm.total).to be < 3
    end
  end
end
