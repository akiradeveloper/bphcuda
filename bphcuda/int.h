#pragma once

#include <bphcuda/tuple3.h>
#include <thrust/tuple.h>

namespace bphcuda{

typedef int Int;
typedef thrust::tuple<Int, Int, Int> Int3;

__host__ __device__
Int3 mk_int3(Int x, Int y, Int z){
  return thrust::make_tuple(x, y, z);
}

} // end of bphcuda
