require "thrusting/build"

module Bphcuda
  class << self
    private 
    def use_bphcuda(cc)
      thisdir = File.expand_path File.dirname __FILE__
      incpath = "#{thisdir}/../.."
      incpath = File.expand_path incpath
      cc.append("-I #{incpath}")
    end
  end
 
  module_function
  def make_default_compiler()
    cc = Thrusting.make_default_compiler()
    cc = use_bphcuda(cc)
    return cc
  end
end

if __FILE__ == $0
  p Bphcuda.make_default_compiler().enable_gtest
end
