#pragma once

#include <bphcuda/tuple3.h>
#include <thrust/tuple.h>

namespace bphcuda{

typedef int Int;
typedef thrust::tuple<Int, Int, Int> Int3;

#include <bphcuda/mk_int.h>

} // end of bphcuda
