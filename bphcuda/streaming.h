#pragma once
namespace bphcuda {

struct move {
  Real3 operator()(Real3 p, Real3 vel, Real dt){
    return p + vel * dt
  }
}

} // end of bphcuda
