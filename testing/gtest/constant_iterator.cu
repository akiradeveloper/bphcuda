#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>

#include <thrusting/list.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>

#include <gtest/gtest.h>

TEST(thrust, transform){
  thrust::transform(
    thrust::make_constant_iterator<int>(1),
    thrusting::advance(2, thrust::make_constant_iterator<int>(1)),
    thrust::make_constant_iterator<int>(1),
    thrust::negate<int>());
}
