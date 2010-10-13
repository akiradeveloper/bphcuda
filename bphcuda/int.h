#pragma once

#include <thrust/tuple.h>

namespace bphcuda{

typedef uint Int;
typedef thrust::tuple<Int, Int, Int> Int3;

Int3 mk_int3(Int x, Int y, Int z){
  return thrust::make_tuple(x, y, z);
}

} // end of bphcuda
