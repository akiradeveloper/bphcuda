#include <thrusting/algorithm/transform.h>
#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <thrust/functional.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

__constant__ int xs[2] = {10,20};

// #define THRUSTING_DEVICE_SPACE

struct constant_filler :public thrust::unary_function<int, int> {
#ifdef THRUSTING_DEVICE_SPACE
  __device__
  int operator()(int x) const {
    return xs[x];
  }
#else
  __host__
  int operator()(int x) const {
    int tmp;
    cudaMemcpyFromSymbol(&tmp, xs, sizeof(int)*1, sizeof(int)*x);      
    return tmp;
  }
#endif
};
  
TEST(CONSTMEMORY, TEST){
  int _xs[3] = {0,1,0};
  vector<int>::type xs(_xs, _xs+3);

  unsigned int a, b;
  a = clock();
  
  thrusting::transform(
    3,
    xs.begin(),
    xs.begin(),
    constant_filler());

  int _ans[3] = {10,20,10};
  vector<int>::type ans(_ans, _ans+3);

  EXPECT_EQ(
    make_list(xs),
    make_list(ans));

  b = clock();
  std::cout << a << std::endl;
  std::cout << (b-a) << std::endl;
}
