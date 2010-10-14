#pragma once

#include <bphcuda/real.h>
#include <bphcuda/distribution.h>
#include <bphcuda/kinetic_e.h>
#include <bphcuda/distribution.h>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

namespace bphcuda {

template<typename Iter>
void relax(Iter ps_first, Iter ps_last, Int seed){
  Real old_kinetic = calc_kinetic_e(ps_first, ps_last);
  alloc_shell_rand(ps_first, ps_last, seed);  
  Real new_kinetic = calc_kinetic_e(ps_first, ps_last);
  Real ratio = old_kinetic / new_kinetic;
  Real3 ratio3 = mk_real3(ratio, ratio, ratio);
  thrust::transform(
    ps_first, ps_last,
    thrust::make_constant_iterator(ratio3),
    ps_first,
    thrust::multiplies<Real3>());
}

// leave it no implemented
template<typename Iter1, typename Iter2>
void share(Iter1 ps_first, Iter1 ps_last, Iter2 ine_first, Int s=0){
}

} // end of bphcuda
