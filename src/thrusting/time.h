#pragma once

#include <iostream>
#include <string>

#include <thrusting/list.h>

#include <vector>

namespace thrusting {

class stopwatch {

  std::string _title;
  cudaEvent_t _start, _end;
  std::vector<float> _times; 

public:

  stopwatch(std::string title)
  :_title(title), _times(0)
  {
    cudaEventCreate(&_start);
    cudaEventCreate(&_end);
  }

  ~stopwatch(){
    cudaEventDestroy(_start);
    cudaEventDestroy(_end);
  }
  
  void begin(){
    cudaEventRecord(_start, 0);
    cudaEventSynchronize(_start);
  }
   
  void end(){
    cudaEventRecord(_end, 0);
    cudaEventSynchronize(_end);

    float elapsed_time;
    /*
      unit is ms
    */
    cudaEventElapsedTime(&elapsed_time, _start, _end);
    _times.push_back(elapsed_time);
  }

  float average(){
    float t = 0;
    for(size_t i=0; i<_times.size(); ++i){
      t += ( _times[i] / _times.size() );
    }    
    return t;
  }
   
  void show(){
    std::cout << "--- " << _title << " ---" << std::endl;
    std::cout << "Average: " << average() << " [ms]" << std::endl;
    std::cout << "Times:" << std::endl;
    std::cout << make_list(_times) << std::endl;
  }
};

} // END thrusting
