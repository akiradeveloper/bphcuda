# This path should be changed by client
THRUST_HOME = "#{ENV["HOME"]}/local/thrust"
THRUST_LIB = THRUST_HOME

thisdir = File.dirname(__FILE__)
LIBPATH = thisdir
VERSION = "0.0.0"
DEVELOPER = ["Akira Hayakawa <ruby.wktk@gmail.com>"]
TESTPATH = 
COMPILER = "nvcc"
CC = [COMPILER, LIBPATH, THRUST_LIB].join " -I "

p CC
