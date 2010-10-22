thisdir = File.dirname(__FILE__)
require [thisdir, "mk_template"].join "/"

if __FILE__ == $0
  $stdout << "#pragma once"
  (2..9).each do |i|
     $stdout << mk_typedef(i, "Real")
  end
  (2..9).each do |i|
     $stdout << mk_template(i, "Real", "mk_real")
  end
end
