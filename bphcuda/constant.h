#pragma once

namespace bphcuda {

__device__
Real AVOGADRO(){
  return 6.02f * __powf(10, 23);
}

__host__ __device__
Real PI(){
  return 3.141592f;
}

} // end of bphcuda
