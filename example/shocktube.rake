require "bphcuda"

cc = Bphcuda.make_default_compiler
#.use_real_precision("double")
cc.deepcopy.append("-m32").append("--use_fast_math").append("-O2").make_compile_task("shocktube")
#cc.deepcopy.make_compile_task("shocktube")

