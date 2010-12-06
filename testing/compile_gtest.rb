thisdir = File.expand_path File.dirname __FILE__ 

require [thisdir, "def_compile"].join "/"

if __FILE__ == $0
  cc = TEST_CC
  cc = add_device_option(cc, "host")
  cc = add_floating_option(cc, "float")
  compile_gtest(cc, "all.test", ARGV)
  system "./all.test"
end
