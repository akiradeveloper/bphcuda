thisdir = File.dirname(__FILE__)
require [thisdir, "..", "project"].join "/"
TESTCC = CC

def compile(src, bin)
  system "#{TESTCC} -o #{bin} #{src}"  
end

def run(bin)
  system "./#{bin}"
end

if __FILE__ == $0
  src = ARGV[0]
  if File.extname(src) == ".cu"
    # ruby compile.rb hoge.cu
    bin = File.basename(src, ".cu") + ".bin"
    compile(src, bin) 
    p "-----COMPILATION FINISHED-----"
    p "-----  START PROCESSING  -----"
    run(bin)
  end
end
