require_relative "common"
require "matrix"
require "kefir"

class Noh2d 

  def initialize(n, m, backend)
    @n = n
    @m = m
    @s = 2
    @backend = backend
  end

  def figurename
    "#{FigureDir}/noh2d_xy_n#{@n}_m#{@m}_s#{@s}_#{@backend}.jpeg"
  end

  def draw
    fname = dirname + "/" + "plot.dat"
    txt = open(fname, "r").read.split("\n")
    size = @m*2
    a = []
    (0...size).each do |i| 
      t = txt[i].split(" ")
      b = []
      (0...size).each do |j|
        b << t[j].to_i
      end
      a << b
    end
    m = Matrix.columns(a)
    Kefir.open() do |gp|
     gp.set 'term', 'jpeg'
     gp.set 'output', figurename.embed
     gp.set 'contour'
     gp.set 'cntrparam', 'bspline'
     gp.set 'cntrparam', 'levels 100'
     gp.unset 'key'
     gp.set 'size', '1,1'
     gp.set 'view', '0,0'
     gp.unset 'surface'
     gp.set 'size', 'square' 
     gp.set 'xlabel', "x".dump    
     gp.set 'ylabel', "y".dump    
     gp.splot do |p|
        p << Kefir.eval(m) do |d|
          d << 'matrix'
          d << 'with lines'
        end
      end 
    end
  end

  def dirname
    "#{DataDir}/noh2d_xy/n#{@n}_m#{@m}_s#{@s}_#{@backend}"
  end

  def binname
    dir = Pathname("../noh2d_xy").realpath
    "#{dir}/main_on_#{@backend}.bin"
  end

  def run
    dir = dirname
    bin = binname
    time = 0.5

    mkdir_p dir
    sh "#{bin} #{n} #{m} #{s} #{time} #{dir}/plot.dat #{dir}/time.dat" 
  end
end

if __FILE__ == $0
  x = Noh2d.new(100,100,"device")
  p x.figurename
  p x.binname
  p x.dirname
  x.draw
end
