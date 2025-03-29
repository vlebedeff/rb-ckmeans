# frozen_string_literal: true

RSpec.describe Ckmeans::Clusterer do
  describe "#new" do
    specify do
      expect { described_class.new(0, 0, []) }.to_not raise_error
    end
  end
end
