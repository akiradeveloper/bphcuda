"""
typedef typename tupleN<real>::type realN
"""

def realN(n)
"""
typedef typename tuple#{n}<real>::type real#{n}
"""
end

if __FILE__ == $0
  for i in 2..9
    $stdout <<  realN(i)
  end
end
