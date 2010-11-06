thisdir = File.expand_path File.dirname __FILE__ 

require [thisdir, "def_compile"].join "/"

GTEST_NAME = "all.test"
def compile_gtest(tests)
  system "#{TESTCC} -o #{GTEST_NAME} all.cu #{tests.join(" ")}"
end

if __FILE__ == $0
  compile_gtest(ARGV)
  run(GTEST_NAME)
end
