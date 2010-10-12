typedef float3 Real3 // conceal Real3 is float3 or float4 for coarescing
typedef float Real // this means nothing in fact

namespace bphcuda {

Real3 mk_real3(Real x, Real y, Real z){
  Real3 p;
  p.x = x;
  p.y = y;
  p.z = z;
}

Real3 operator+(Real3 p1, Real3 p2){
  Real3 p;
  p.x = p1.x + p2.x;
  p.y = p1.y + p2.y;
  return p;
}

} // end of bphcuda
