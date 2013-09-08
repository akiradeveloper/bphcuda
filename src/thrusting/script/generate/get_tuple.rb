module Thrusting
  private 
  def get_tuple(i, tuple)
    "thrust::get<#{i}>(#{tuple})"
#"""
#thrust::get<#{i}>(#{tuple})
#"""
  end
end

if __FILE__ == $0
  include Thrusting
  print get_tuple(1, "x")
end
