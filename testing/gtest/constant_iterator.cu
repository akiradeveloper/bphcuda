#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>

#include <thrusting/list.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/vector.h>

#include <gtest/gtest.h>

TEST(thrust, transform){
  size_t n_particle = 2;
  THRUSTING_VECTOR<int> output(n_particle);
  thrust::transform(
    thrust::make_constant_iterator<int>(1),
    thrusting::advance(n_particle, thrust::make_constant_iterator<int>(1)),
    output.begin(),
    // thrust::make_constant_iterator<int>(1), // will fail. can not alloc on constant_iterator
    thrust::negate<int>());
}
