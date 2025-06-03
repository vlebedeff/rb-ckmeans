# frozen_string_literal: true

RSpec.describe Ckmedian::Clusterer do
  describe "#clusters" do
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

      example do
        expect(described_class.new(x, 1, 8).clusters).to eq([x])
      end
    end
  end
end
