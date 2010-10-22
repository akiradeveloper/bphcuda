"""
N : LENGTH 
op : Operator
template<typename T0, typename T1, ...>
thrust::tuple<T0, T1, ...> operator op (thrust::tuple<T0, T1, ...> x, thrust::tuple<T0, T1, ...> y){
  return thrust::make_tuple(x.get<0>() op y.get<0>(), ...);
}
"""
  
def operator(n, op)
arg = (0...n*3).map { |i| "typename T#{i}" }
arg2 = (0...n).map { |i| "T#{i}" }
arg3 = (n...n*2).map { |i| "T#{i}" }
arg4 = (n*2...n*3).map { |i| "T#{i}" }
input = (0...n).map { |i| "x.get<#{i}>()#{op}y.get<#{i}>()" }
"""
template<#{arg.join(", ")}>
thrust::tuple<#{arg4.join(", ")}> operator op (const thrust::tuple<#{arg2.join(", ")}> &x, const thrust::tuple<#{arg3.join(", ")}> &y){
  return thrust::make_tuple(#{input.join(", ")});
}
"""
end

"""
template<typename T0, typename T1, ...>
std::stream &operator<<(const ostream &os, thrust::tuple<T0, T1, ...> x){
}
"""

def ostream(n)
arg = (0...n).map { |i| "typename T#{i}" }
arg2 = (0...n).map { |i| "T#{i}" }
s = (0...n).map { |i| "x.get<#{i}>()" }.join("+ ',' +")
"""
template<#{arg.join(", ")}>
std::stream &operator<<(const ostream &os, const thrust::tuple<#{arg2.join(", ")}> &x){
  std::string s;
  s = '(' + #{s} + ')';
  os << s;
  return s;
}
"""
end

"""
template<typename T0, ...>
bool operator==(const thrust::tuple<T0, ...> x, const thrust::tuple<T0, ...> y){
  return x.get<0>() == y.get<1>() && ...;
}
bool operator!=
  return ! (x==y);
}
"""

def equality(n)
end


if __FILE__ == $0
  print operator(3, "+")
  print ostream(3)
end
