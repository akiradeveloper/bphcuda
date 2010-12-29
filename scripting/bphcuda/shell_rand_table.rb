def make_shell_rand(z, angle)
  cs = z
  sn = Math.sqrt(1 - cs*cs)
  b = angle
  cx = sn * Math.sin(b)
  cy = sn * Math.cos(b)
  cz = cs
  return [cx, cy, cz]
end

def to_real3(ary3)
  return "real3(#{ary3[0]},#{ary3[1]},#{ary3[2]})"
end

def make_stride_array(s, e, sep)
  step = (e-s).to_f / sep
  ary = []
  x = s
  while x <= e
    ary << x
    x += step
  end
  return ary
end

def make_all_combination(ary1, ary2)
  ary = [] 
  ary1.each do |x| 
    ary2.each do |y|
      ary << [x, y]
    end
  end
  ary
end

if __FILE__ == $0
  sep_z = 10
  sep_angle = 10
  zs = make_stride_array(-1, 1, sep_z)
  angles = make_stride_array(0, 2*Math::PI, sep_angle)

  #p make_all_combination(zs, angles)
  shell_table = make_all_combination(zs, angles)
  .map { |z, angle| to_real3(make_shell_rand(z, angle))  }
  .join ","

  txt =
  """
  __constant__
  real3 SHELL_TABLE[] = { #{shell_table} };
  """ 
  $stdout << txt
end
