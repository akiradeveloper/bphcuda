"""
op : op charcter for example '+'
functor : for example 'plus'

void operator(op)=(A with){
  typedef typename thrust::iterator_value<A>::type VALUE_TYPE;
  thrust::transform(head, head+n, with, head, thrust::(functor)<VALUE_TYPE>());
}
"""


def _parallel_operator(op, functor_name)
"""
void operator#{op}=(A with){
  typedef typename thrust::iterator_value<A>::type VALUE_TYPE;
  thrust::transform(head, head+n, with, head, thrust::#{functor_name}<VALUE_TYPE>());
}
"""
end

if __FILE__ ==$0
  print _parallel_operator('+', "plus")
  op = ['+', '-', '*', '/', '%', '&', '|', '^'] 
  functor = ["plus", "minus", "multiplies", "divides", "modulus", "bit_and", "bit_or", "bit_xor"]
  p op.zip(functor)
  op.zip(functor).each do |op, functor|
    $stdout << _parallel_operator(op, functor)
  end
end  
