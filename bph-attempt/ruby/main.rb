def x()
  return 10
end

def y()
  return 10
end

def z()
  return 10
end

def cellx()
  lenx = 1.0
  return lenx / x
end

def celly()
  leny = 1.0
  return leny / y
end

def cellz()
  lenz = 1.0
  return lenz / z
end

def n_g()
  return x * y * z
end

def n()
  return 5 * 5 * 5
end

def weight()
  return 1.0
end

class Float3
  def initialize(x,y,z)
    @x = x
    @y = y
    @z = z
  end
  attr_reader :x,:y,:z
  def plus(with)
    return Float3.new(@x+with.x, @y+with.y, @z+with.z)
  end
end

def initialize_point()
  ary = []
  5.times do |i|
    5.times do |j|
      5. times do |z|
         interval = 0.1
         ary << Float3.new(0.1*i, 0.1*j, 0.1*z)
      end
    end
  end
  return ary.map { |p| p.plus(Float3.new(0.3, 0.3, 0.3)) }
end

def convert2index1D(index3D)
  return y()*z()*index3D.x + z()*index3D.y + index3D.z
end

def detect_cell_index(p)
  return Float3.new(
    (p.x / cellx()).floor,
    (p.y / celly()).floor,
    (p.z / cellz()).floor)
end

  
if $0 == __FILE__
  d_array_point_position = initialize_point()
  d_array_point_belonging = Array.new(n()){0}
  d_array_cell_begin = Array.new(n_g()){0}
  d_array_cell_contain = Array.new(n_g()){0}
  def time()
    return 100
  end
  def dt()
    return 0.01
  end
  def do_calc_belong()
    d_array_point_position.times do |i|
      d_array_point_belonging[i] =
        convert2index1D(detect_cell_index(d_array_point_position[i]))
    end
  end
  def do_calc_contain()
    d_array_point_belonging.each do |ind|
      d_array_cell_contain[ind] += 1
    end
  end
  def do_prefix_sum()
    def sum(to)
      s = 0
      for i in 0..to
        s += d_array_cell_contain[i]
      end
    end
    d_array_cell_begin.size.times do |ind|
      d_array_cell_begin[ind] = sum()
    end
  end
  def do_move()
    def move_point_seq(from, len)
      def move(i)
        f = Float3.new(0,0,0)
        for ind in from...from+len
          unless ind == i
            f = f.plus(force(i, ind))
          end
        end
        
      end
      for i in from...form+len
        move(i)
      end
    end
      
    d_array_cell_begin.zip(d_array_cell_contain).do |from, len|
      move_point_seq(from, len)
    end
  end
      
  def force(p1, p2)
    def distance()
      return Math.sqrt( (p1.x-p2.x)**2 + (p1.y-p1.y)**2 )
    end
    return 1.0 / distance()**2
  end
  
  
  def      

  time().times do
    calc()
    print(d_array_point_position)
  end
end
