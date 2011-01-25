require_relative "common"

class Wallshock 
  
  def initialize(n, m, backend)
    @n = n
    @m = m
    @s = 2
    @backend = backend
  end

  def figurename
    "#{FigureDir}/wallshock_x_n#{@n}_m#{@m}_s#{@s}_#{@backend}.jpeg"
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
      gp.set('output', figurename.embed)
      gp.set('xrange', '[0:0.3]')
      gp.set('yrange', '[0.5:6.5]')

      gp.plot do |p|
        p << Kefir.eval([xs, ys]) do |d|
          d << 'with lines'
        end
        p << Kefir.eval(analytic_data.embed) do |d|
          d << 'using 2:3'
          d << 'title "analytic"'
          d << 'with lines'
        end
      end
    end
  end

  def analytic_data
    "#{AnalyticDir}/wallShock_gm140.dat"
  end

  def dirname
    "#{DataDir}/wallshock_x/n#{@n}_m#{@m}_s#{@s}_#{@backend}"
  end
  
  def binname
    "wallshock_x/main_on_#{@backend}.bin"
  end
  
  def run
    dir = dirname
    bin = binname
    time = 0.5
    task dir => bin do |t|
      mkdir_p dir
      sh "#{bin} #{n} #{m} #{s} #{time} #{dir}/plot.dat #{dir}/time.dat" 
    end
  end
end

if __FILE__ == $0
  x = Wallshock.new(4000,1000,"device")
  p x.analytic_data
  p x.binname
  p x.dirname
  x.draw
end
