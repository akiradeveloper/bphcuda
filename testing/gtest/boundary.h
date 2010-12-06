#pragma once

#include <bphcuda/boundary.h>
#include <thrusting/real.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

TEST(Boundary, Mirroring){
  EXPECT_EQ(10, make_mirroring_functor(7)(4));
  EXPECT_EQ(4, make_mirroring_functor(7)(10));
}

TEST(Boundary, RetrieveGreater){
  EXPECT_EQ(6, make_retrieve_greater_functor(0, 10)(16));
  EXPECT_EQ(6, make_retrieve_greater_functor(0, 10)(6));
}

TEST(Boundary, RetrieveLess){
  EXPECT_EQ(6, make_retrieve_less_functor(0, 10)(-4));
  EXPECT_EQ(6, make_retrieve_less_functor(0, 10)(6));
}
