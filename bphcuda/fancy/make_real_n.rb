"""
realN make_realN(real x1, real x2, ...){
  return make_tupleN<real>(x1, x2, ...);
}
"""

def make_realN(n)
arg = (0...n).map { |i| "real x#{i}" }
input = (0...n).map { |i| "x#{i}" }
"""
real#{n} make_real#{n}(#{arg.join(", ")}){
  return make_tuple#{n}<real>(#{input.join(", ")});
}
"""
end  
  

if __FILE__ == $0
  for i in 2..9
    $stdout << make_realN(i)
  end
end
  
