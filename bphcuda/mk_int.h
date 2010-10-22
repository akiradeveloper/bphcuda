#pragma once
typedef thrust::tuple<Int, Int> Int2;

typedef thrust::tuple<Int, Int, Int> Int3;

typedef thrust::tuple<Int, Int, Int, Int> Int4;

typedef thrust::tuple<Int, Int, Int, Int, Int> Int5;

typedef thrust::tuple<Int, Int, Int, Int, Int, Int> Int6;

typedef thrust::tuple<Int, Int, Int, Int, Int, Int, Int> Int7;

typedef thrust::tuple<Int, Int, Int, Int, Int, Int, Int, Int> Int8;

typedef thrust::tuple<Int, Int, Int, Int, Int, Int, Int, Int, Int> Int9;

__host__ __device__
Int2 mk_int2(Int x1, Int x2){
  return thrust::make_tuple(x1, x2);
}

__host__ __device__
Int3 mk_int3(Int x1, Int x2, Int x3){
  return thrust::make_tuple(x1, x2, x3);
}

__host__ __device__
Int4 mk_int4(Int x1, Int x2, Int x3, Int x4){
  return thrust::make_tuple(x1, x2, x3, x4);
}

__host__ __device__
Int5 mk_int5(Int x1, Int x2, Int x3, Int x4, Int x5){
  return thrust::make_tuple(x1, x2, x3, x4, x5);
}

__host__ __device__
Int6 mk_int6(Int x1, Int x2, Int x3, Int x4, Int x5, Int x6){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6);
}

__host__ __device__
Int7 mk_int7(Int x1, Int x2, Int x3, Int x4, Int x5, Int x6, Int x7){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6, x7);
}

__host__ __device__
Int8 mk_int8(Int x1, Int x2, Int x3, Int x4, Int x5, Int x6, Int x7, Int x8){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6, x7, x8);
}

__host__ __device__
Int9 mk_int9(Int x1, Int x2, Int x3, Int x4, Int x5, Int x6, Int x7, Int x8, Int x9){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6, x7, x8, x9);
}
