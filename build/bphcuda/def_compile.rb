["def_compile"].each do |f|
  require "build/thrusting/#{f}"
end

["bphcuda"].each do |f|
  require "build/path/#{f}"
end

cc = THRUSTING_CXX

cc = use_bphcuda(cc)

BPHCUDA_CXX = cc
