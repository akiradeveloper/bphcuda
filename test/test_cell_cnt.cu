#include "util.h"

#include <bphcuda/cell_cnt.h>

using namespace bphcuda;

int main(void){
  Real xs[3] = {1,1,2};
  thrust::device_vector<Real> d_sample(xs, xs+3);
  thrust::device_vector<Real> d_hist(3);
  mk_cell_cnt(d_sample.begin(), d_sample.end(), d_hist.begin(), d_hist.end());
  Real ans[3] = {0,2,1};
  thrust::device_vector<Real> d_ans(ans, ans+3);
  ASSERT_TRUE(d_hist == d_ans);
  return 0;
}
