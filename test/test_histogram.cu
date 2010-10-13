#include "util.h"
#include <bphcuda/histogram.h>
#include <thrust/device_vector.h>

int main(void){
  Real xs[3] = {1,1,2};
  thrust::device_vector<Real> d_sample(xs, xs+3);
  thrust::device_vector<Real> d_hist(3);
  bphcuda::mk_histogram(d_sample.begin(), d_sample.end(), d_hist.begin(), d_hist.end());
  Real ans[3] = {0,2,1};
  thrust::device_vector<Real> d_ans(ans, ans+3);
  CHECK_EQUAL_ARRAY(d_hist, d_ans);
  return 0;
}
