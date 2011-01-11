require "kefir"

if __FILE__ == $0
  
  fname = ARGV[0]
  
  txt = open(fname, "r").read.split("\n")
  N = 480
  a = []
  (0...N).each do |i| 
    t = txt[i].split(" ")
    # p t
    b = []
    (0...N).each do |j|
      b << t[j].to_i
    end
    a << b
  end

  m = Matrix.columns(a)

# Kefir.open() do |gp|
#    gp.unset 'surface'
#    gp.set 'contour'
#    gp.set 'view', '0,0'
#    gp.set 'table', '"table.dat"'
#    gp.splot do |plot|
#      plot << Kefir.eval(m) do |d|
#        d << 'matrix'
#      end
#    end
#  end

  Kefir.open() do |gp|
    gp.splot do |plot|
      #gp.set 'contour'
      #gp.set 'cntrparam', 'bspline'
      #gp.set 'cntrparam', 'levels 100'
      #gp.set 'cntrparam', 'levels incremental 0, 1, 100'
      gp.unset 'key'
      gp.set 'view', '90,0'
      #gp.unset 'surface'
      #gp.set 'cntrparam', 'levels incremental 0, 1, 10'
      gp.set 'term', 'jpeg'
      gp.set 'output', '"hoge.jpeg"'
      gp.set 'size', 'square'
      plot << Kefir.eval(m) do |d|
        d << 'matrix'
        d << 'with lines'
      end
    end 
  end
end
