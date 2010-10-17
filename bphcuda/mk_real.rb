thisdir = File.dirname(__FILE__)
require [thisdir, "mk_template"].join "/"

if __FILE__ == $0
  (1..9).each do |i|
     $stdout << mk_template("mk_real", "Real", i)
  end
end
