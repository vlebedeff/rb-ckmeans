# frozen_string_literal: true

RSpec.describe Ckmeans do
  it "has a version number" do
    expect(Ckmeans::VERSION).not_to be nil
  end

  it "does something useful" do
    expect(true).to eq(true)
  end

  specify do
    expect(Ckmeans.c_do_nothing).to eq(nil)
  end
end
