#pragma once 

#include <thrusting/time.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(StopWatch, TestByLooping){
  thrusting::stopwatch w("stopwatch test");
  
  for(size_t i=0; i<5; ++i){
    w.begin();
    int x = 0;
    for(size_t j=0; j<1000000; ++j){
      x++;
    }
    w.end();
  }
     
  EXPECT_TRUE(w.average() > 0);
  
  w.show();
}
