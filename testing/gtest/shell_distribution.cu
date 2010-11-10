#include <thrusting/vector.h>

#include <bphcuda/distribution/shell_distribution.h>

#include <gtest/gtest.h>

TEST(shell_distribution, printout){
  size_t count = 10000;
  thrust::device_vector<real3> output(count);
  alloc_shell_rand(output.begin(), output.end(), 0);
  // std::cout << output << std::endl;
}
