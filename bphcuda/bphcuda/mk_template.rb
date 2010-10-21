# typedef Real3 thrust::tuple<Real, Real, Real>
def mk_typedef(n, type)
types = (1..n).map { |i| type }
"""
typedef thrust::tuple<#{types.join(", ")}> #{type}#{n};
"""
end

# ["x1", "x2" .."xn"]
def mk_params(n, type)
  return (1..n).map { |i| "x#{i}" }
end

def mk_template(n, type, name)
"""
__host__ __device__
#{type}#{n} #{name}#{n}(#{mk_params(n, type).map { |x| "#{type} #{x}"}.join(", ")}){
  return thrust::make_tuple(#{mk_params(n, type).join (", ")});
}
"""
end

if __FILE__ == $0
  p mk_typedef(10, "Real")
  p mk_params(10, "Real")
  p mk_template(10, "Real", "mk_real")
end
