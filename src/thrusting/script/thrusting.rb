require "rake/clean"

["get_tuple", "make_autogen_task", "namespace.rb", "template_type", "tuple_min_max"].each do |f|
  require "generate/#{f}"
end

module Thrusting
  class << self
    include Thrusting
  end
  module_function
  def private_module_function(*names)
    names.each do |name|
      module_function name
      private_class_method name
    end
  end
end
