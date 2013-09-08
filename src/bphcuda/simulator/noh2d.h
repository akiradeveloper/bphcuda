namespace bphcuda {
  
  real3 get_center_of(cell &c){
    real x = real(0.5) * (c.x_min() + c.x_max());
    real y = real(0.5) * (c.y_min() + c.y_max());
    real z = real(0.5) * (c.z_min() + c.z_max());

    return real3(x, y, z);
  }

  real get_normal_velocity(real3 &p, real3 &toward){
    real3 vec = p - toward;
    real len = get_distance(p, toward);
    return vec / len;
  }

  real get_distance(real3 &p, real3 & q){
    real x = get<0>(p) - get<0>(q);
    real y = get<1>(p) - get<1>(q);
    real z = get<2>(p) - get<2>(q);
    return thrusting::sqrtr(x*x + y*y + z*z);
  }
}
