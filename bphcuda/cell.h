struct Cell {
  Real3 origin;
  Int3 divisions;
  Real3 spaces;
}
  
Cell mk_cell(Real3 origin, Int3 divisions, Real3 spaces){
  Cell c;
  c.origin = origin;
  c.divisions = divisions;
  c.spaces = spaces;
}

Int3 calc_ind3(const Cell &c, const Real3 p){
}

Int conv_ind3_ind1(const Cell &c, const &ind3){
}
