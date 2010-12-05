thisdir = File.expand_path File.dirname __FILE__ 

["def_compile"].each do |s|
  require "bphcuda/rb/#{s}"
end

# gtest is 32bit
gtest_home = "#{ENV["HOME"]}/local/gtest/gtest-1.5.0"
gtest_lib = [gtest_home, "lib"].join "/"
gtest_include = [gtest_home, "include"].join "/"

cc = CC 
cc += " -I #{gtest_include}"
cc += " -L #{gtest_lib}"
cc += " -l gtest"
cc += " -g"

TEST_CC = cc

def compile_gtest(cc, bin, files)
  sources = ["gtest_main.cu"] + files
  compile(cc, bin, sources)
end
