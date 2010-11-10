#pragma once

#include <bphcuda/cell.h>

/*
  A class that draw given cell and particles in it.
*/
struct particle_drawer {
  cell _cell;
  particle_drawer(cell &c)
  :_cell(c){}

  void init_gl() const {
  }

  void draw_cell() const {
  }

  void draw_particle() const {
  }
   
  template<typename R>
  void draw(
    size_t n_particle, 
    R x, R y, R z
  ){
    draw_cell();
    draw_particles(n_particle, x, y, z);
  }
