"""
template<typename T>
struct tupleN {
  typedef thrust::tuple<T, T, ...> type;
};
"""

def tupleN(n)
arg = (0...n).map { "T" }
"""
template<typename T>
struct tuple#{n} {
  typedef thrust::tuple<#{arg.join(", ")}> type;
};
"""
end

if __FILE__ == $0
  for i in 2..9
    $stdout << tupleN(i)
  end
end
