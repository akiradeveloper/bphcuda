#pragma once

namespace bphcuda{

typedef uint Int;
typedef tuple<uint, uint, uint> Int3;

Int3 mk_int3(Int x, Int y, Int z){
  return thrust::make_tuple(x, y, z);
}

} // end of bphcuda
