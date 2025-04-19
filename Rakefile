# frozen_string_literal: true

require "bundler/gem_tasks"
require "rspec/core/rake_task"
require "rake/extensiontask"

Rake::ExtensionTask.new("extensions") do |ext|
  ext.lib_dir = "lib/ckmeans"
  ext.ext_dir = "ext/ckmeans"
end

RSpec::Core::RakeTask.new(:spec)

require "rubocop/rake_task"

RuboCop::RakeTask.new

task default: %i[compile spec rubocop]
