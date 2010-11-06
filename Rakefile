thisdir = File.expand_path File.dirname __FILE__

task :push do
  rep = "http://bitbucket.org/akiradeveloper/bphcuda"
  sh "hg push #{rep}"
end

task :remove_deprecated do
  `hg status`.split("\n").grep(/^!/).each do |x|
    sh "hg remove #{x.split.at(1)}"
  end
end
