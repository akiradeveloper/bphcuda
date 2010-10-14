#include "util.h"

#include <bphcuda/real.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>

using namespace bphcuda;

int main(void){
  Real3 r1 = mk_real3(0.0, 1.0, 2.0);
  Real3 r2 = mk_real3(3.0, 4.0, 5.0);
  
  Real3 r3 = r1 + r2;
  ASSERT_TRUE(r3 == mk_real3(3.0, 5.0, 7.0));

  Real3 r4 = r1 * r2;
  ASSERT_TRUE(r4 == mk_real3(0.0, 4.0, 10.0));

  // real3 * real only correct and real * real3 is not correct
  // if the value is int it runs correctly
  Real3 r5 = r2 * 2; 
  // std::cout << r5.get<0>() << r5.get<1>() << r5.get<2>() << std::endl;
  ASSERT_EQUAL(r5, mk_real3(6.0, 8.0, 10.0));
  
  // inversed order: real3 * real -> real * real3
  Real3 r6 = 2 * r2;
  ASSERT_EQUAL(r6, mk_real3(6.0, 8.0, 10.0));
  
  // FOR list
  thrust::device_vector<Real3> vec;
  vec.push_back(r1);
  thrust::device_vector<Real3> out(1);
  thrust::transform(vec.begin(), vec.end(), out.begin(), thrust::identity<Real3>());
  ASSERT_EQUAL(out, vec);
  
  Real3 sum = thrust::reduce(
    thrust::make_transform_iterator(vec.begin(), thrust::identity<Real3>()),
    thrust::make_transform_iterator(vec.end(), thrust::identity<Real3>()),
    mk_real3(0.0,0.0,0.0),
    thrust::plus<Real3>());
  ASSERT_EQUAL(sum, r1);

  return 0;
}
