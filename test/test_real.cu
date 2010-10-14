#include "util.h"

#include <bphcuda/real.h>

using namespace bphcuda;

int main(void){
  Real3 r1 = mk_real3(0.0, 1.0, 2.0);
  Real3 r2 = mk_real3(3.0, 4.0, 5.0);
  
  Real3 r3 = r1 + r2;
  ASSERT_TRUE(r3 == mk_real3(3.0, 5.0, 7.0));

  Real3 r4 = r1 * r2;
  ASSERT_TRUE(r4 == mk_real3(0.0, 4.0, 10.0));

  
  return 0;
}
