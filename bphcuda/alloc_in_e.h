#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/transform.h>

#include <thrusting/iterator.h>
#include <thrusting/real.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/functional.h>

#include <bphcuda/kinetic_e.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

template<typename Real1, typename Real2, typename Int>
void alloc_in_e(
  size_t n_particle, 
  Real1 u, Real1 v, Real1 w,
  Real2 m,
  Real1 in_e, // output
  real s
  Int idx,
  size_t n_cell,
  Real1 tmp1,
  Int tmp2
){
  thrust::fill(
    in_e,
    thrusting::advance(n_particle, in_e),
    real(0));
   
  thrust::pair<Int, Real> end;
  /*
  */
  end = thrust::reduce_by_bucket(
    idx,
    thrusting::advance(n_particle, idx),
    thrust::make_transform_iterator(
      thrusting::make_zip_iterator(u, v, w, m),
      kinetic_e_calculator()),
    tmp2,
    tmp1);

  /*
  */
  thrust::transform();
   
  /*
  */
  thrust::transform();

  /*
  */
  thrust::gather 
}

} // END bphcuda
