require "gp"
require "matrix"

class Noh2d 

  def initialize(name)
    @name = name
  end

  def draw
    fname = "plot/#{@name}.dat"
    txt = open(fname, "r").read.split("\n")
    size = txt.size
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
    GP.open() do |gp|
     gp.set 'term', 'jpeg'
     gp.set 'output', "figure/#{@name}.jpeg".dump
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
        p << GP.eval(m) do |d|
          d << 'matrix'
          d << 'with lines'
        end
      end 
    end
  end
end
