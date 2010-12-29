require "bphcuda"

cc = Bphcuda.make_default_compiler
cc.deepcopy.append("--use_fast_math").append("-O3").make_compile_task("shocktube_x")
#cc.deepcopy.make_compile_task("shocktube")

