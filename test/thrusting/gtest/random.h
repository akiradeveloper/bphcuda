#pragma once 

#include <thrust/random.h>

#include <gtest/gtest.h>

/*
  Chech if all random engine
  are equipped with
  a) constructor with seed
  b) discard method
*/
TEST(Random, RandomGenerator){
  int seed = 777;
  int step = 777;

  thrust::ranlux24 r24(seed);
  r24.discard(step);

  thrust::ranlux48 r48(seed);
  r48.discard(step);

  thrust::taus88 t88(seed);
  t88.discard(step);

  thrust::default_random_engine d(seed);
  d.discard(step);
}
