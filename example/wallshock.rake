require "bphcuda"

cc = Bphcuda.make_default_compiler
cc.deepcopy.append(Thrusting::DEFAULT_OPTIMIZE_FLAG).make_compile_task("wallshock")
#cc.deepcopy.make_compile_task("shocktube")

