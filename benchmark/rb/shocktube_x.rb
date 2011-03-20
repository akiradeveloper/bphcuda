require_relative "common"

class Shocktube

  def initialize(n, m, s=2, backend)
    @n = n
    @m = m
    @s = s
    @backend = backend
  end

  def analytic_data
    "#{AnalyticDir}/shtb_g140_t010t020.dat"
  end

  def dirname
    dir = "#{DataDir}/shocktube_x/n#{@n}_m#{@m}_s#{@s}_#{@backend}"
  end
  
  def binname
    dir = Pathname("../shocktube_x")
    "#{dir.realpath}/main_on_#{@backend}.bin"
  end
  
  def run
    dir = dirname
    bin = binname
    # according to isaka's experiment, t is 0.15
    time = 0.15
#    task dir => bin do |t|
      mkdir_p dir
      sh "#{bin} #{@n} #{@m} #{@s} #{time} #{dir}/plot.dat #{dir}/time.dat"
#    end
  end
 
  def figurename
    "#{FigureDir}/shocktube_x_n#{@n}_m#{@m}_s#{@s}_#{@backend}.jpeg"
  end

  def plotname
    "#{FigureDir}/shocktube_x_n#{@n}_m#{@m}_s#{@s}_performance.jpeg"
  end

  def draw
    f = open( dirname + "/" + "plot.dat" )
    content = f.read.split("\n")
    ys = content
    size = content.size
    step = 1.0 / size
    xs = (0...size).map { |i| i*step + 0.5*step }
    
    Kefir.open do |gp|
      gp.set('terminal', 'jpeg')
      gp.set('xlabel', 'x'.dump)
      gp.set('ylabel', 'density'.dump)
      outputname = figurename
      p outputname
      gp.set('output', outputname.embed)
      gp.plot do |p|
        p << Kefir.eval([xs, ys]) do |d|
          title = "computed on #{devicename(@backend)}" 
          d << "title #{title.dump}"
          d << 'with lines'
        end
        p << Kefir.eval(analytic_data.embed) do |d|
          d << 'using 2:4'
          d << 'title "analytic"'
          d << 'with lines'
        end
      end
    end
  end
end

if __FILE__ == $0
  x = Shocktube.new(500,1000,"device")
  p x.analytic_data
  p x.binname
  p x.dirname
  x.draw
end
