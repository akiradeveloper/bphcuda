#include "util.h"

#include <bphcuda/cell_prefix.h>

int main(void){
  Int[3] xs = {0,2,5};
  thrust::device_vector<Int> d_xs(xs, xs+3);

  thrust::device_vector<Int> d_out(3);
  bphcuda::mk_prefix_sum(d_xs.begin(), d_xs.end(), d_out);

  Int[3] ans = {0,0,2};
  thrust::device_vector<Int> d_ans(xs, xs+3);
  CHECK_EQUAL_ARRAY(d_out, d_ans);
  return 0;
} 
