#include <thrust/iterator/constant_iterator.h>

#include <thrusting/dtype/real.h>
#include <thrusting/vector.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/list.h>

#include <bphcuda/distribution/maxwell_distribution.h>

#include <iostream>

#include <gtest/gtest.h>

namespace {
  using thrusting::real;
  using namespace thrusting::op;
}

TEST(maxwell_distribution, printout){
  size_t count = 3;
  THRUSTING_VECTOR<real> u(count);
  THRUSTING_VECTOR<real> v(count);
  THRUSTING_VECTOR<real> w(count);
  real T = 1.0;
  size_t seed = 0;
  bphcuda::alloc_maxwell_rand(
    count,
    u.begin(), v.begin(), w.begin(), 
    thrust::make_constant_iterator<real>(1.0),
    T,
    seed);
  std::cout << 
    thrusting::make_list(count, thrusting::make_zip_iterator(u.begin(), v.begin(), w.begin())) 
  << std::endl;
}
