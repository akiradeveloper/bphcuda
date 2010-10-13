#pragma once

#include <bphcuda/real.h>
#include <bphcuda/distribution.h>
#include <bphcude/kinetic_e.h>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

namespace bphcuda {

    
template<typename Iter>
void relax(Iter xs_first, Iter xs_last){
  Real old_kinetic = calc_kinetic_e(xs_first, xs_last);
  
  Real new_kinetic = calc_kinetic_e(xs_first, xs_last);
}

} // end of bphcuda
