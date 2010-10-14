#include "util.h"

#include <bphcuda/kinetic_e.h>

using namespace bphcuda;

int main(void){
  Real3 r = mk_real3(1.0, 2.0, 3.0);
  Real result = kinetic_e()(r); 	
  ASSERT_EQUAL(result, 14.0);  
  return 0;
}
