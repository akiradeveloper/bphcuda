class Sjogreen

  def initialize(name)
    @name = name
  end

  def draw
    f = open("plot/#{@name}.dat" )
    content = f.read.split("\n")
    ys = content
    size = content.size
    step = 1.0 / size
    xs = (0...size).map { |i| i*step + 0.5*step }
    
    GP.open do |gp|
      gp.set('terminal', 'jpeg')
      gp.set('output', "figure/#{@name}.jpeg".dump)
      gp.set('xrange', '[0:0.9]')
      gp.set('yrange', '[-0.1:1.1]')
      gp.set('xlabel', 'x'.dump)
      gp.set('ylabel', 'density'.dump)
      gp.plot do |p|
        p << GP.eval([xs, ys]) do |d|
          d << "title #{@name.dump}"
          d << 'with lines'
        end
      end
    end
  end
end
