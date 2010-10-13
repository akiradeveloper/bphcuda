#include "util.h"

#include <bphcuda/cell_prefix.h>

using namespace bphcuda;

int main(void){
  Int xs[3] = {0,2,5};
  thrust::device_vector<Int> d_xs(xs, xs+3);

  thrust::device_vector<Int> d_out(3);
  mk_cell_prefix(d_xs.begin(), d_xs.end(), d_out.begin());

  Int ans[3] = {0,0,2};
  thrust::device_vector<Int> d_ans(ans, ans+3);
  ASSERT_TRUE(d_out == d_ans);
  return 0;
} 
