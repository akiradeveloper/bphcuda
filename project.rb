# This path should be changed by client
THRUST_LIB = "#{ENV["HOME"]}/local/thrust"

thisdir = File.dirname(__FILE__)
LIBPATH = [thisdir, "bphcuda"].join "/"
VERSION = "0.0.0"
DEVELOPER = ["Akira Hayakawa <ruby.wktk@gmail.com>"]
TESTPATH = 
COMPILER = "nvcc"
CC = [COMPILER, LIBPATH, THRUST_LIB].join " -I "
