require "bphcuda"

cc = Bphcuda.make_default_compiler
cc.deepcopy.use_real_precision("double").append("--use_fast_math").append("-O2").make_compile_task("shocktube_x")
#cc.deepcopy.make_compile_task("shocktube")

