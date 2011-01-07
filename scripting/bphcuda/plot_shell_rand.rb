thisdir = File.expand_path File.dirname __FILE__

require "#{thisdir}/shell_rand_table"
require "kefir"

if __FILE__ == $0
  include Bphcuda
  tab =  make_shell_table(10, 10)
  x = []
  y = []
  z = []
  tab.each do |a|
    x << a[0]
    y << a[1]
    z << a[2]
  end

  p x.size
  p y.size
  p z.size
  #p [x, y, z]
  
  Kefir.open do |gp|
    gp << 'set terminal jpeg'
    gp << 'set output "shell_rand_by_map.jpeg"'   
    gp << 'set view 90,90'
    gp.splot do |p|
      p << Kefir.eval([x, y, z]) do |d|
        d << 'title "shellrand"'
      end
    end
  end
end
