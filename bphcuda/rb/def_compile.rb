# describing overall configuration of compilation

thisdir = File.expand_path File.dirname __FILE__ 

libpath = [thisdir, "..", ".."].join "/"

thrusting_home = "#{ENV["HOME"]}/sandbox/thrusting" 
thrusting_include = thrusting_home

thrust_home = "#{ENV["HOME"]}/local/thrust"
thrust_include = thrust_home

cuda_home = "/usr/local/cuda"
cuda_lib = [cuda_home, "lib"].join "/"
  
cc = "nvcc"
cc = [cc, thrusting_include, thrust_include, libpath].join " -I"
cc += " -L #{cuda_lib}"

cc += " -Xcompiler -trigraphs"

CC = cc

def add_device_option(cc, name)
  case name 
  when "host"
    cc
  when "device"
    cc += " -D THRUSTING_USING_DEVICE_VECTOR"
  when "omp" 
    cc += " -D THRUSTING_USING_DEVICE_VECTOR"
    cc += " -Xcompiler -fopenmp"
    cc += " -D THRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP"
  else
    raise "invalid name"
  end
end

def add_floating_option(cc, name)
  case name
  when "double"
    cc += " -D THRUSTING_USING_DOUBLE_FOR_REAL"
  when "float"
    cc
  else
    raise "invalid name"
  end
end

def compile(cc, bin, files)
  p cc
  system "#{cc} -o #{bin} #{files.join(" ")}"
end
