#include <bphcuda/real.h>

#include <thrust/transform_reduce.h>

struct class kinetic_e :public thrust::unary_function<Real3, Real> {
  __host__ __device__
  Real operator()(const Real3 &x){
    Real3 p = x*x;
    return p.x + p.y + p.z;
  }
};

template<typename Iter>
Real calc_kinetic_e(Iter cs_first, Iter cs_last){
  return transform_reduce(cs_first, cs_last, kinetic_e(), 0, thrust::plus<Real>);
}
