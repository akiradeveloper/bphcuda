#include <thrust/iterator/constant_iterator.h>

#include <thrusting/dtype/real.h>
#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/iterator/zip_iterator.h>

#include <bphcuda/distribution/shell_distribution.h>

#include <gtest/gtest.h>

namespace {
  using thrusting::real;
  using namespace thrusting::op;
}

TEST(shell_distribution, printout){
  size_t count = 3;
  THRUSTING_VECTOR<real> u(count);
  THRUSTING_VECTOR<real> v(count);
  THRUSTING_VECTOR<real> w(count);
  size_t seed = 0;
  bphcuda::alloc_shell_rand(
    count,
    u.begin(), v.begin(), w.begin(), 
    seed);
  std::cout << 
    thrusting::make_list(count, thrusting::make_zip_iterator(u.begin(), v.begin(), w.begin())) 
  << std::endl;
}
