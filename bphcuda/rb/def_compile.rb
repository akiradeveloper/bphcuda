# describing overall configuration of compilation

thisdir = File.expand_path File.dirname __FILE__ 

LIBPATH = [thisdir, "..", ".."].join "/"

THRUST_HOME = "#{ENV["HOME"]}/local/thrust"
THRUST_INCLUDE = THRUST_HOME

THRUSTING_HOME = "#{ENV["HOME"]}/sandbox/thrusting"
THRUSTING_INCLUDE = THRUSTING_HOME

CUDA_HOME = "/usr/local/cuda"
CUDA_LIB = [CUDA_HOME, "lib"].join "/"

cc = "nvcc"
cc = [cc, THRUSTING_INCLUDE, THRUST_INCLUDE, LIBPATH].join " -I"
cc = [cc, CUDA_LIB].join " -L"
cc = [cc, "-Xcompiler -trigraphs"].join " "
# cc = [cc, "-D THRUSTING_USING_DEVICE_VECTOR"].join " " 
# cc = [cc, "-D THRUSTING_USE_DOUBLE_FOR_REAL"].join " "
CC = cc
