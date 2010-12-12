def use_bphcuda(cc)
  thisdir = File.expand_path File.dirname __FILE__
  bph_include = "#{thisdir}/../../../.."
  cc += " -I #{bph_include}" 
end
