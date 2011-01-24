require "kefir"

def plot1d(dir)
  f = open( "data/" + dir + "/" + "plot.dat" )
  content = f.read.split("\n")
  ys = content
  size = content.size
  p size
  step = 1.0 / size
  xs = (0...size).map { |i| i*step + 0.5*step }
  # p xs
  
  Kefir.open do |gp|
    gp.set('xrange', '[0:1]')
    gp.set('terminal', 'jpeg')
    outputname = dir + ".jpeg"
    p outputname
    gp.set('output', outputname.embed)
    gp.plot do |p|
      p << Kefir.eval([xs, ys]) do |d|
        d << 'with lines'
      end
    end
  end
end

def plot_shocktube(dir, out)
end

def plot_wallshock(dir, out)
end

def plot_sjogreen(dir, out)
end

if __FILE__ == $0
  dir = ARGV[0]
  plot1d( dir ) 
end
