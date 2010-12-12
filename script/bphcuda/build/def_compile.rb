["def_compile"].each do |f|
  require "thrusting/build/#{f}"
end

["bphcuda"].each do |f|
  require "bphcuda/build/path/#{f}"
end

cc = THRUSTING_CXX

cc = use_bphcuda(cc)

BPHCUDA_CXX = cc
