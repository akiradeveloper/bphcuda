#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>

#include <iostream>
void assert(bool cond) { std::cout << "FALSE" << std::endl; }
//template<typename Iter>
//void puts(Iter first, Iter last){
//  std::cout << "[";
//  int size = last - first;
//  for(int i=0; i<size; i++){ std::cout << *(first+i) << ","; }
//  std::cout << "]" << std::endl;
//}

typedef float Real;
typedef thrust::tuple<Real, Real, Real> Real3;
 
Real3 make_real3(Real x, Real y, Real z){
  return thrust::make_tuple(x, y, z);
}

Real3 operator+(const Real3 &self, const Real3 &with){
  Real x = self.get<0>() + with.get<0>();
  Real y = self.get<1>() + with.get<1>();
  Real z = self.get<2>() + with.get<2>();
  return make_real3(x, y, z);
}
  
int main(void){
  Real3 a = make_real3(1.0, 2.0, 3.0);

  // addition works. 
  Real3 b = a + a; 

  // transform works too though ...
  thrust::device_vector<Real3> vec;
  vec.push_back(a);
  thrust::device_vector<Real3> out(1);
  thrust::transform(vec.begin(), vec.end(), out.begin(), thrust::identity<Real3>());
  
  // can not compile, why ...
  Real3 sum = thrust::reduce(vec.begin(), vec.end(), make_real3(0.0, 0.0, 0.0));
  return 0;
}
