module Bphcuda
  class << self
    include Bphcuda
  end

  def make_shell_table(nzs, nangle)
    sep_z = nzs - 1
    sep_angle = nangle
    zs = make_stride_array(-1, 1, sep_z)
    angles = make_stride_array(0, 2*Math::PI, sep_angle)
    angles = angles[0, angles.size-1]

    shell_table = make_all_combination(zs, angles)
    .map { |z, angle| make_shell_rand(z, angle)  }

    shell_table
  end

  def make_shell_rand_map(nzs, nangle)
    shell_table = make_shell_table(nzs, nangle)
  
    n = nzs * nangle
    txt =
"""
real SHELL_TABLE_X[#{n}] = { 
#{shell_table.map {|x| x[0]}.join(", ")} 
};

real SHELL_TABLE_Y[#{n}] = {
#{shell_table.map {|x| x[1]}.join(", ")}
};

real SHELL_TABLE_Z[#{n}] = {
#{shell_table.map {|x| x[2]}.join(", ")}
};
""" 
    txt
  end

  private
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
end

if __FILE__ == $0
  include Bphcuda
  print make_shell_rand_map(10, 10) 
end
