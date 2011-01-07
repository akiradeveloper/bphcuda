require "thrusting"

["shell_rand_table"].each do |f|
  require "bphcuda/#{f}"
end

module Bphcuda
  class << self
    include Bphcuda
  end

  public
  def make_default_compiler()
    cc = Thrusting.make_default_compiler()
    use_bphcuda(cc)
    return cc
  end

  private 
  def use_bphcuda(cc)
    thisdir = File.expand_path File.dirname __FILE__
    incpath = "#{thisdir}/.."
    incpath = File.expand_path incpath
    cc << "-I #{incpath}"
  end
end

if __FILE__ == $0
  include Bphcuda

  cxx = make_default_compiler
  cxx.enable_debug_mode(true)
  p cxx
end
