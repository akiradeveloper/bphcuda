namespace bphcuda {

template<typename Iter>
void sort_by_ind1(Iter xs_first, Iter xs_last, Cell& c){
}

template<typename OrderedIter, typename OutputIter>
void mk_histogram(
  OrderedIter sample_fisrt, OrderedIter sample_last, 
  OutputIter histo_first, OutputIter histo_last
){
}

template<typename Iter>
void mk_prefixsum(
  Iter histo_first, Iter histo_last, 
  Iter prefix_first
){
}

} // end of bphcuda
