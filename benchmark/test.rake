namespace "test" do
task "wallshock_x" do
  sh "rake wallshock_x/main_on_device.bin"
  sh "wallshock_x/main_on_device.bin 100 1000 0 0.5 wallshock_x_plot.dat wallshock_x_time.dat"
end

task "shocktube_x" do
  sh "rake shocktube_x/main_on_device.bin"
  sh "shocktube_x/main_on_device.bin 100 500 0 0.16 shocktube_x_plot.dat shocktube_x_time.dat"
end

task "sjogreen_x" do
  x = "sjogreen_x"
  sh "rake #{x}/main_on_device.bin"
  sh "#{x}/main_on_device.bin 100 1000 0 5 0.1 #{x}_plot.dat #{x}_time/dat"
end

task "noh2d_xy" do
  
end
end
