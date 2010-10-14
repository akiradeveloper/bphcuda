thisdir = File.dirname(__FILE__)
require [thisdir, "..", "project"].join "/"
TESTCC = CC

def compile(src, bin)
  system "#{TESTCC} -o #{bin} #{src}"  
end

if __FILE__ == $0
  input = ARGV[0]
  if File.extname(input) == ".cu"
    # ruby compile.rb hoge.cu
    compile(input, File.basename(input, ".cu"))
  end
end
