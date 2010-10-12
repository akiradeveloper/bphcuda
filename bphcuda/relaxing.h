#include <bphcuda/value.h>
#include <bphcuda/distribution.h>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

namespace bphcuda {

struct class kinetic_e :public thrust::unary_function<Real3, Real> {
  Real operator()(const Real3 &x){
    Real3 p = x*x;
    return p.x + p.y + p.z;
  }
};
    
template<typename Iter>
void relax(Iter xs_first, Iter xs_last){
  Real old_kinetic = transform_reduce(xs_first, xs_last, kinetic_e(), 0, thrust::plus<Real>);
  
  Real new_kinetic = transform_reduce(xs_first, xs_last, kinetic_e(), 0, thrust::plus<Real>);
}
} // end of bphcuda
