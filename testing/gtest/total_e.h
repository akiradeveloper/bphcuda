#pragma once

#include <bphcuda/total_e.h>

#include <thrusting/real.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

TEST(TotalE, Test){
  EXPECT_EQ(real(10), make_total_e_calculator()(real5(1,1,1,2,7)));
}
