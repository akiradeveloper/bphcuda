require "bphcuda"

cc = Bphcuda.make_default_compiler
cc.make_compile_task("shocktube")
