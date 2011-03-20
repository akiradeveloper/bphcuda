["common", "shocktube_x", "sjogreen_x", "wallshock_x", "noh2d_xy"].each do |f|
  require_relative f
end

class Performance

  class Shocktube

    def initialize(n)
      @n = n 
    end
   
    def figurename
      "#{PerformanceDir}/shocktube_x_n#{@n}.jpeg"
    end
 
    def total_time(m, backend)
      TimeData.new(::Shocktube.new(@n, m, backend).dirname).total_time(backend)
    end

    def draw(ms, devices=Devices)
      Kefir.open() do |gp|
        gp.set("terminal", "jpeg")
        gp.set("output", figurename.dump)
        gp.set('xlabel', "M".dump)
        gp.set("ylabel", "time [ms]".dump)
        gp.plot do |p| 
          devices.each do |backend|
            ys = ms.map { |m| total_time(m, backend) }
            p << Kefir.eval([ms, ys]) do |d|
              d << "title #{devicename(backend).dump}"
              d << "with linespoints"
            end
          end
        end
      end
    end
  end
 
  
  class Wallshock

    def initialize(n)
      @n = n 
    end
   
    def figurename
      "#{PerformanceDir}/wallshock_x_n#{@n}.jpeg"
    end
 
    def total_time(m, backend)
      TimeData.new(::Wallshock.new(@n, m, backend).dirname).total_time(backend)
    end

    def draw(ms, devices=Devices)
      Kefir.open() do |gp|
        gp.set("terminal", "jpeg")
        gp.set("output", figurename.dump)
        gp.set('xlabel', "M".dump)
        gp.set("ylabel", "time [ms]".dump)
        gp.plot do |p| 
          devices.each do |backend|
            ys = ms.map { |m| total_time(m, backend) }
            p << Kefir.eval([ms, ys]) do |d|
              d << "title #{devicename(backend).dump}"
              d << "with linespoints"
            end
          end
        end
      end
    end
  end

  class Sjogreen
    def initialize(n)
      @n = n 
    end
   
    def figurename
      "#{PerformanceDir}/sjogreen_x_n#{@n}.jpeg"
    end
 
    def total_time(m, backend)
      TimeData.new(::Sjogreen.new(@n, m, backend).dirname).total_time(backend)
    end

    def draw(ms, devices=Devices)
      Kefir.open() do |gp|
        gp.set("terminal", "jpeg")
        gp.set("output", figurename.dump)
        gp.set('xlabel', "M".dump)
        gp.set("ylabel", "time [ms]".dump)
        gp.plot do |p| 
          devices.each do |backend|
            ys = ms.map { |m| total_time(m, backend) }
            p << Kefir.eval([ms, ys]) do |d|
              d << "title #{devicename(backend).dump}"
              d << "with linespoints"
            end
          end
        end
      end
    end
  end 

  class Noh2d

    def initialize(n)
      @n = n 
    end
   
    def figurename
      "#{PerformanceDir}/noh2d_xy_n#{@n}.jpeg"
    end
 
    def total_time(m, backend)
      TimeData.new(::Noh2d.new(@n, m, backend).dirname).total_time(backend)
    end

    def draw(ms, devices=Devices)
      Kefir.open() do |gp|
        gp.set("terminal", "jpeg")
        gp.set("output", figurename.dump)
        gp.set('xlabel', "M".dump)
        gp.set("ylabel", "time [ms]".dump)
        gp.plot do |p| 
          devices.each do |backend|
            ys = ms.map { |m| total_time(m, backend) }
            p << Kefir.eval([ms, ys]) do |d|
              d << "title #{devicename(backend).dump}"
              d << "with linespoints"
            end
          end
        end
      end
    end
  end
end

if __FILE__ == $0
  x = Performance::Shocktube.new(500)
  x.draw([1,10,100,1000])

  x = Performance::Wallshock.new(4000)
  x.draw([1,10,100,1000])

  x = Performance::Sjogreen.new(4000)
  x.draw([1,10,100,1000])

  x = Performance::Noh2d.new(100)
  x.draw([1,10,100])
end
