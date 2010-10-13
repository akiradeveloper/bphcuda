#include "util.h"

#include <bphcuda/assert.h>

int main(void){
  ASSERT_TRUE(1==1);
  int xs[2] = {1,2}; thrust::device_vector<int> d_xs(xs, xs+2);
  int ys[2] = {1,2}; thrust::device_vector<int> d_ys(ys, ys+2);
  int zs[2] = {1,3}; thrust::device_vector<int> d_zs(zs, zs+2);
  int ws[3] = {1,2,3}; thrust::device_vector<int> d_ws(ws, ws+3);
  ASSERT_TRUE(d_xs == d_ys);
  // ASSERT_TRUE(v_xs == v_zs);
  // ASSERT_TRUE(v_xs == v_ws);
  return 0;
}
