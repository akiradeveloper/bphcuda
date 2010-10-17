# ["x1", "x2" .. ]
def mk_params(n)
  return (1..n).map { |i| "x#{i}" }
end

def mk_template(name, type, n)
"""
#{type}#{n} #{name}#{n}(#{mk_params(n).map { |x| "#{type} #{x}"}.join(", ")}){
  return thrust::make_tuple(#{mk_params(n).join (", ")});
}
"""
end
