#pragma once

#include <thrusting/assert.h>

#include <gtest/gtest.h>

TEST(Assert, Test){
  THRUSTING_CHECK(10==10);
}
