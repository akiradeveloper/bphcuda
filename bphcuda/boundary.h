#include <thrusting/functional.h>
#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>

#include <bphcuda/cell.h>

namespace {
  using thrusting::real;
  using thrusting::real6;
}

namespace bphcuda {
  
bool is_out_cell_plus_x(const cell &c, real x){
}

bool is_out_cell_minus_x(const cell &c, real x){
} 

bool is_out_cell_plus_y(const cell &c, real y){
}

bool is_out_cell_minus_y(const cell &c, real y){
}

bool is_out_cell_plus_z(const cell &c, real z){
}

bool is_out_cell_minus_z(const cell &c, real z){
}

struct is_out_cell_plus_x :public thrust::unary_function<real6, bool>{
}

} // END bphcuda
