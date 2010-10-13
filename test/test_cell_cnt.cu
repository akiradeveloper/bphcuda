#include "util.h"

#include <bphcuda/cell_cnt.h>

using namespace bphcuda;

int main(void){
  Int xs[3] = {1,1,2};
  thrust::device_vector<Int> d_sample(xs, xs+3);
  thrust::device_vector<Int> d_hist(3);
  mk_cell_cnt(d_sample.begin(), d_sample.end(), d_hist.begin(), d_hist.end());
  Int ans[3] = {0,2,1};
  thrust::device_vector<Int> d_ans(ans, ans+3);
  ASSERT_TRUE(d_hist == d_ans);
  return 0;
}
