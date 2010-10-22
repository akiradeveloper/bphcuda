"""
template<typename T>
tupleN<T>::type operator*(tupleN<T>::type x, T n){
  return make_tupleN<T>(x.get<0>() * n, x.get<1>() * n, ...);
}
tupleN<T>::type operator*(T n, tupleN<T>::type x){
  return x * n;
}
tupleN<T>::type operator/(tupleN<T>::type x, T n){
  return make_tupleN<T>(x.get<0>() / n, ...);
}
"""

def _operator(n, op)
input = (0...n).map { |i| "x.get<#{i}>()#{op}y" }
"""
template<typename T>
tuple#{n}<T>::type operator*(tuple#{n}<T>::type x, T y){
  return make_tuple#{n}<T>(#{input.join(", ")});
}
"""
end

def operator(n)
"""
#{_operator(n, "*")}
tupleN<T>::type operator*(T n, tupleN<T>::type x){
  return x * n;
}
#{_operator(n, "/")}
"""
end
  
if __FILE__ == $0
  print operator(3)
end
