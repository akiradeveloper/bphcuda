namespace "test" do

task "wallshock_x" do
  sh "rake wallshock_x/main_on_device.bin"
  sh "wallshock_x/main_on_device.bin 4000 1000 2 0.5 wallshock_x_plot.dat wallshock_x_time.dat"
end

task "shocktube_x" do
  sh "rake shocktube_x/main_on_device.bin"
  sh "shocktube_x/main_on_device.bin 500 1000 2 0.16 shocktube_x_plot.dat shocktube_x_time.dat"
end

task "sjogreen_x" do
  x = "sjogreen_x"
  sh "rake #{x}/main_on_device.bin"
  sh "#{x}/main_on_device.bin 4000 1000 2 0.1 5 #{x}_plot.dat #{x}_time.dat"
end

task "noh2d_xy" do |t|
  x = "noh2d_xy" 
  sh "rake #{x}/main_on_device.bin"
  sh "#{x}/main_on_device.bin 100 100 2 0.5 #{x}_plot.dat #{x}_time.dat"
  sh "ruby plot_noh2d.rb initial_noh2d_xy.dat"
  mv "hoge.jpeg", "initial_noh2d_xy.jpeg"
  sh "ruby plot_noh2d.rb #{x}_plot.dat"
end

task "all" => ["wallshock_x", "shocktube_x", "sjogreen_x", "noh2d_xy"]
end
