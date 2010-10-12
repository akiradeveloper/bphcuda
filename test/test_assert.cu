#include "bphcuda/assert.h"
#include <vector>

int main(void){
  CHECK(1==1, "equal value");
  // CHECK(1==2, "not equal value");
  CHECK_EQUAL_VALUE(1, 1);
  // CHECK_EQUAL_VALUE(1, 2);
  float3 a; a.x=10, a.y=20; a.z=30;
  float3 b; b.x=10, b.y=20; b.z=30;
  float3 c; c.x=10, c.y=20; c.z=31;
  CHECK_EQUAL_VALUE3(a, b);
  // CHECK_EQUAL_VALUE3(a, c);
  int xs[2] = {1,2}; std::vector<int> v_xs(xs, xs+2);
  int ys[2] = {1,2}; std::vector<int> v_ys(ys, ys+2);
  int zs[2] = {1,3}; std::vector<int> v_zs(zs, zs+2);
  int ws[3] = {1,2,3}; std::vector<int> v_ws(ws, ws+3);
  CHECK_EQUAL_ARRAY(v_xs, v_ys);
  // CHECK_EQUAL_ARRAY(v_xs, v_zs);
  // CHECK_EQUAL_ARRAY(v_xs, v_ws);
  
  return 0;
}
