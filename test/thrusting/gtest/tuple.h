#pragma once

#include <thrusting/tuple.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/vectorspace.h>

#include <iostream>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Tuple, Plus){
  // T + T
  EXPECT_EQ(thrusting::make_tuple2(3,5), thrusting::make_tuple2(2,3) + thrusting::make_tuple2(1,2));
} 

TEST(Tuple, Minus){
  // T - T
  EXPECT_EQ(thrusting::make_tuple2(1,1), thrusting::make_tuple2(2,3) - thrusting::make_tuple2(1,2));
}

TEST(Tuple, Multiplies){
  // v * T
  EXPECT_EQ(thrusting::make_tuple2(2,2), 2 * thrusting::make_tuple2(1,1));
}

TEST(Tuple, Divides){
  // T / v
  EXPECT_EQ(thrusting::make_tuple2(1,1), thrusting::make_tuple2(2,2) / 2);
}

TEST(Tuple, MakeString){
  EXPECT_EQ("(1, 2)", thrusting::detail::make_string(thrusting::make_tuple2(1,2)));
}
  
TEST(Tuple, Ostream){
  std::cout << thrusting::make_tuple2(1,2) << std::endl;
}

TEST(Tuple, ArithmaticPriority){
  tuple2<int>::type x = thrusting::make_tuple2(1,2);
  tuple2<int>::type y = thrusting::make_tuple2(3,4);
  
  EXPECT_EQ(thrusting::make_tuple2(7, 10), x + 2 * y);
  EXPECT_EQ(thrusting::make_tuple2(8, 12), 2 * (x + y));
}
