
Int1 mk_int1(Int x1){
  return thrust::make_tuple(x1);
}

Int2 mk_int2(Int x1, Int x2){
  return thrust::make_tuple(x1, x2);
}

Int3 mk_int3(Int x1, Int x2, Int x3){
  return thrust::make_tuple(x1, x2, x3);
}

Int4 mk_int4(Int x1, Int x2, Int x3, Int x4){
  return thrust::make_tuple(x1, x2, x3, x4);
}

Int5 mk_int5(Int x1, Int x2, Int x3, Int x4, Int x5){
  return thrust::make_tuple(x1, x2, x3, x4, x5);
}

Int6 mk_int6(Int x1, Int x2, Int x3, Int x4, Int x5, Int x6){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6);
}

Int7 mk_int7(Int x1, Int x2, Int x3, Int x4, Int x5, Int x6, Int x7){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6, x7);
}

Int8 mk_int8(Int x1, Int x2, Int x3, Int x4, Int x5, Int x6, Int x7, Int x8){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6, x7, x8);
}

Int9 mk_int9(Int x1, Int x2, Int x3, Int x4, Int x5, Int x6, Int x7, Int x8, Int x9){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6, x7, x8, x9);
}
