"""
template<typename T>
typename tupleN<T>::type make_tupleN(T x0, T x1, ...){
  return thrust::make_tuple(x0, x1, ...);
}
"""

def tupleN(n)
arg = (0...n).map { |i| "T x#{i}" }
input = (0...n).map { |i| "x#{i}" }
"""
template<typename T>
typename tuple#{n}<T>::type make_tuple#{n}(#{arg.join(", ")}){
  return thrust::make_tuple(#{input.join(", ")});
}
"""
end

if __FILE__ == $0
  print tupleN(3)
end
