#pragma once

namespace bphcuda {

__host__ __device__
Real AVOGADRO(){
  return 6.02F * powf(10, 23);
}

__host__ __device__
Real PI(){
  return 3.141592F;
}

} // end of bphcuda
