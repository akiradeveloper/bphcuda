require_relative "common"

class Sjogreen

  def initialize(n, m, backend)
    @n = n
    @m = m
    @s = 2
    @backend = backend
  end

  def figurename
    "#{FigureDir}/sjogreen_x_n#{@n}_m#{@m}_s#{@s}_#{@backend}.jpeg"
  end

  def dirname
    "#{DataDir}/sjogreen_x/n#{@n}_m#{@m}_s#{@s}_#{@backend}"
  end
  
  def analytic_data
    "#{AnalyticDir}/sjgm140u7t01.dat"
  end
  
  def binname
    "sjogreen_x/main_on_#{@backend}.bin"
  end
  
  def run
    dir = dirname
    bin = binname
    # with 7 as u_0 vacuum occurs 
    u_0 = 7
    # in accordance with isaka's experiments
    time = 0.1
    task dir => bin do |t|
      mkdir_p dir
      sh "#{bin} #{n} #{m} #{s} #{time} #{u_0} #{dir}/plot.dat #{dir}/time.dat" 
    end
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
      gp.set('xrange', '[0:0.9]')
      gp.set('yrange', '[-0.1:1.1]')
      gp.plot do |p|
        p << Kefir.eval([xs, ys]) do |d|
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
  x = Sjogreen.new(4000,1000,"device")
  p x.analytic_data
  p x.binname
  p x.dirname
  p x.figurename
  x.draw
end
