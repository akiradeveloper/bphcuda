#pragma once

#include <thrust/tuple.h>

namespace bphcuda{

// long -> still bug for transform_reduce that shows "int" in errmsg
typedef int Int;
typedef thrust::tuple<Int, Int, Int> Int3;

__host__ __device__
Int3 mk_int3(Int x, Int y, Int z){
  return thrust::make_tuple(x, y, z);
}

} // end of bphcuda
