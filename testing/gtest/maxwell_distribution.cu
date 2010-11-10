#include <thrusting/dtype/real.h>
#include <thrusting/vector.h>

#include <bphcuda/distribution/maxwell_distribution.h>

#include <gtest/gtest.h>

TEST(maxwell_distribution, printout){
  Int count = 10;
  thrust::device_vector<Real3> output(count);
  Real T = 1;
  Real m = 1;
  alloc_maxwell_rand(output.begin(), output.end(), 0, T, m);
  std::cout << output << std::endl;
}
