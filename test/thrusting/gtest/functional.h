#pragma once

#include <thrusting/functional.h>
#include <thrusting/tuple.h>
#include <thrusting/vector.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Functional, Flip){
  // 4 / 2 = 2
  int x = thrusting::flip(thrust::divides<int>())(2, 4);
  EXPECT_EQ(2, x);
}

TEST(Functional, Multiplies){
  int x = thrusting::multiplies<int, int>()(2, 2);
  EXPECT_EQ(4, x);
}

TEST(Functional, Divides){
  int x = thrusting::divides<int, int>()(4, 2);
  EXPECT_EQ(2, x);
}

TEST(Functional, LeftShift){
  int x = thrusting::left_shift<int, int>()(2, 2);
  EXPECT_EQ(8, x);
}

TEST(Functional, RightShift){
  int x = thrusting::right_shift<int, int>()(8, 2);
  EXPECT_EQ(2, x);
}

TEST(Functional, Bind1st){
  // 2 / 1 = 2
  int x = thrusting::bind1st(thrust::divides<int>(), 2)(1);
  EXPECT_EQ(2, x);
}

TEST(Functional, Bind2nd){
  // 4 / 2 = 2
  int x = thrusting::bind2nd(thrusting::divides<long, int>(), 2)(4L);
  EXPECT_EQ(2L, x);
}

TEST(Functional, Bind2nd1){
  bool b = thrusting::bind2nd(thrust::greater<size_t>(), 1)(1);
  EXPECT_FALSE(b);
}

TEST(Functional, Bind2nd2){
  bool b = thrusting::bind2nd(thrust::greater<size_t>(), 1)(2);
  EXPECT_TRUE(b);
}

struct sum_f :public thrust::unary_function<thrust::tuple<int, int>, int> {
  __host__ __device__
  int operator()(const thrust::tuple<int, int> &t) const {
    return thrust::get<0>(t) + thrust::get<1>(t);
  }
};

TEST(FunctionalTest, SumF){
  EXPECT_EQ(5, sum_f()(thrust::make_tuple(2,3)));
}

TEST(Functional, Curry){
  // 2 + 3 = 5
  int x = thrusting::curry(sum_f())(2,3);
  EXPECT_EQ(5, x);
}

TEST(Functional, UnCurry){
  // 2 + 3 = 5
  int x = thrusting::uncurry(thrust::plus<int>())(thrust::make_tuple(2,3));
  EXPECT_EQ(5, x);
}

TEST(Functional, Compose){
  // -1 -> (-) -> () -> 1
  EXPECT_EQ(1, thrusting::compose(thrust::negate<int>(), thrust::identity<int>())(-1));
}

TEST(Functional, Compose2){
  // 1 -> (+1) -> (*2) -> 4
  int x= 
    thrusting::compose(
      thrusting::bind1st(thrust::multiplies<int>(), 2),
      thrusting::bind1st(thrust::plus<int>(), 1))(1);
  EXPECT_EQ(4, x);
}
