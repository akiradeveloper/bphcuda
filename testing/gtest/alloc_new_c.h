#pragma once

#include <bphcuda/relax_cell.h>

#include <thrusting/real.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(RelacCell, AllocNewC){
  /*
    the last element is zero speed
  */
  EXPECT_EQ(
    real3(0.0, 0.0, 0.0),
    thrusting::iterator_value_at(2, thrusting::make_zip_iterator(us.begin(), vs.begin(), ws.begin())));
}
