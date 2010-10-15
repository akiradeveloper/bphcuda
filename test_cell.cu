#include <bphcuda/cell.h>

#include "util.h"

using namespace bphcuda;

int main(void){
  Cell c = mk_cell(
    mk_real3(0,0,0),
    mk_real3(1,1,1),
    mk_int3(2,2,2));
  
  ASSERT_EQUAL(calc_ind1(c, mk_real3(1.5,1.5,1.5)), 7);
  ASSERT_EQUAL(calc_ind1(c, mk_real3(0.5,0.5,0.5)), 0);

  return 0;
}
