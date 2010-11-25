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
  R1 u, R1 v, R1 w,
  I1 cell_idx,
  size_t n_cell,
  R1 ave_u, R1 ave_v, R1 ave_w,
  I1 cell_cnt
){
}
  
void plus_average_velocity(
  size_t n_particle,
  R1 u, R1 v, R1 w,
  I1 cell_idx,
  size_t n_cell,
  R1 ave_u, R1 ave_v, R1 ave_w,
  I1 cell_cnt
){
}
  
} // END bphcuda
