# configuration for google test

thisdir = File.expand_path File.dirname __FILE__ 

["def_compile"].each do |s|
  require "thrusting/rb/#{s}"
end

# gtest is 32bit
GTEST_HOME = "#{ENV["HOME"]}/local/gtest/gtest-1.5.0"
GTEST_LIB = [GTEST_HOME, "lib"].join "/"
GTEST_INCLUDE = [GTEST_HOME, "include"].join "/"

testcc = CC 
testcc = [testcc, GTEST_INCLUDE].join " -I"
testcc = [testcc, GTEST_LIB].join " -L"
testcc = [testcc, "gtest"].join " -l"
testcc = [testcc, "-g"].join " "
# testcc = [testcc, "-D THRUSTING_USING_DEVICE_VECTOR"].join " " 

TESTCC = testcc

p TESTCC

def run(bin)
  system "./#{bin}"
end
