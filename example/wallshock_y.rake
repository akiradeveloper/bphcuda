require "bphcuda"

cc = Bphcuda.make_default_compiler
cc.deepcopy.use_real_precision("double").append(Thrusting::DEFAULT_OPTIMIZE_FLAG).make_compile_task("wallshock_y")
#cc.deepcopy.make_compile_task("shocktube")

