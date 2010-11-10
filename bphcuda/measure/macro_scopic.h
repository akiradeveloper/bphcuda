#pragma once

/*
  A class that measure the macro scopic values at a cell.
*/
struct macro_scopic {
  cell _cell;
  macro_scopic(cell &c)
  :_cell(c){}

  real T(size_t i, size_t j, size_t k){
  }
  
  real rho(size_t i, size_t j, size_t k){
  }
  
  real pressure(){
  }
};
