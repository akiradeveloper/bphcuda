require "thrusting/build"

module Bphcuda
  class << self
    private 
    def use_bphcuda(cc)
      thisdir = File.expand_path File.dirname __FILE__
      bph_include = "#{thisdir}/../.."
      bph_include = File.expand_path bph_include
      cc += " -I #{bph_include}" 
    end
  end
 
  module_function
  def make_default_compiler(cc)
    cc = Thrusting.make_default_compiler(cc)
    cc = use_bphcuda(cc)
    return cc
  end
end
