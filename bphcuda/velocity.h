#pragma once

#include <thrust/gather.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

void minus_average_velocity(
  size_t n_particle,
  R1 u, R1 v, R1 w, // input and output
  I1 cell_idx, // input
  size_t n_cell,
  R1 ave_u, R1 ave_v, R1 ave_w, // output
  IntTmp tmp
){
  // reduce_by_key -> (cnt, value)
  thrust::reduce_by_key(
    cell_idx,
    thrusting::advance(n_particle, cell_idx),
    thrusting::make_zip_iterator(u, v, w),
    tmp, // cnt
    thrusting::make_zip_iterator(ave_u, ave_v, ave_w));
    
  // value / cnt
  thrust::transform(
    thrusting::make_zip_iterator(ave_u, ave_v, ave_w),
    thrusting::advance(n_cell, thrusting::make_zip_iterator(ave_u, ave_v, ave_w)),
    tmp,
    thrusting::make_zip_iterator(ave_u, ave_v, ave_w),
    thrusting::divides<real3, real>()); // TODO crach if division is 0
    
  // minus 
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrusting::make_permutation_iterator(
      thrusting::make_zip_iterator(ave_u, ave_v, ave_w),
      cell_idx),
    thrusting::make_zip_iterator(u, v, w),
    thrust::minus<real3>());
}
  
void plus_average_velocity(
  size_t n_particle,
  R1 u, R1 v, R1 w,
  I1 cell_idx,
  size_t n_cell,
  R1 ave_u, R1 ave_v, R1 ave_w
){
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrusting::make_permutation_iterator(
      thrusting::make_zip_iterator(ave_u, ave_v, ave_w),
      cell_idx),
    thrusting::make_zip_iterator(u, v, w),
    thrust::plus<real3>());
}

} // END bphcuda
