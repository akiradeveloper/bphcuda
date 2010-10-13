#include <bphcuda/assert.h>
#include <bphcuda/real.h>
#include <bphcuda/cell.h>
#include <bphcuda/macro.h>
#include <bphcuda/int.h>
#include <thrust/device_vector.h>

Cell test_cell(){
  return mk_cell(
    mk_real3(0.0, 0.0, 0.0),
    mk_real3(1.0, 1.0, 1.0),
    mk_int3(2,2,2));
}

