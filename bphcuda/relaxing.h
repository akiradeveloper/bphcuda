#pragma once

#include <bphcuda/real.h>
#include <bphcuda/distribution.h>
#include <bphcude/kinetic_e.h>
#include <bphcuda/distribution.h>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

namespace bphcuda {

template<typename Iter>
void relax(Iter xs_first, Iter xs_last){
  Real old_kinetic = calc_kinetic_e(xs_first, xs_last);
  alloc_shell_rands(xs_first, xs_last);  
  Real new_kinetic = calc_kinetic_e(xs_first, xs_last);
  Real ratio = old_kinetic / new_kinetic;
  thrust::transform(
    xs_first, xs_last,
    thrust::make_constant_iterator(ratio),
    xs_first,
    thrust::multiplies<Real3>());
}

// leave it no implemented
template<typename Iter1, typename Iter2>
void share(Iter1 ps_first, Iter1 ps_last, Iter2 ine_first, Int s=0){
}

} // end of bphcuda
