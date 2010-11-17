#pragma once

#include <thrusting/dtype/real.h>

namespace {
  using thrusting::real;
}

namespace bphcuda {

__host__ __device__
real GRAVITATIONAL_CONST(){
  return 6.6728e-11;
}

__host__ __device__
real BOLTZMANN(){
  return 1.38e-23;
}

__host__ __device__
real AVOGADRO(){
  return 6.02e23;
}

__host__ __device__
real PI(){
  return 3.141592;
}

} // END bphcuda
