#include <thrust/sort.h>

#include <thrusting/list.h>
#include <thrusting/vector.h>
#include <thrusting/vectorspace.h>
#include <thrusting/real.h>
#include <thrusting/functional.h>
#include <thrusting/pp.h>
#include <thrusting/time.h>

#include <thrusting/algorithm/transform.h>
#include <thrusting/algorithm/reduce_by_bucket.h>
#include <thrusting/algorithm/remove.h>
#include <thrusting/algorithm/sort.h>

#include <bphcuda/cell.h>
#include <bphcuda/bph.h>
#include <bphcuda/boundary.h>
#include <bphcuda/streaming.h>
#include <bphcuda/force.h>
#include <bphcuda/random/uniform_random.h>
#include <bphcuda/alloc_in_e.h>

namespace {
  using namespace thrust;
  using namespace thrusting;
  using namespace bphcuda;
}

namespace bphcuda {

  __host__ __device__
  real3 get_center_of(cell &c){
    real x = real(0.5) * (c.x_min() + c.x_max());
    real y = real(0.5) * (c.y_min() + c.y_max());
    real z = real(0.5) * (c.z_min() + c.z_max());

    return real3(x, y, z);
  }

  __host__ __device__
  real get_distance(const real3 &p, const real3 & q){
    real x = get<0>(p) - get<0>(q);
    real y = get<1>(p) - get<1>(q);
    real z = get<2>(p) - get<2>(q);
    return thrusting::sqrtr(x*x + y*y + z*z);
  }

  __host__ __device__
  real3 get_normal_velocity(const real3 &p, const real3 &toward){
    real3 vec = toward - p;
    real len = get_distance(p, toward);
    return vec / len;
  }

  class alloc_normal_velocity_functor :public thrust::unary_function<real3, real3> {
    real3 _center;
  public:
    alloc_normal_velocity_functor(real3 center)
    :_center(center){};
    __host__ __device__
    real3 operator()(const real3 &p) const {
      return get_normal_velocity(p, _center);
    }
  };

  class is_out_of_circle :public thrust::unary_function<real3, bool> {
    real3 _center;
    real _radius;
  public:
    is_out_of_circle(real3 center, real radius)
    :_center(center), _radius(radius){};
    __host__ __device__
    bool operator()(const real3 &p){
      real distance = get_distance(_center, p);
      return distance > _radius;
    }
  };

  class is_out_of_circle_2 :public thrust::unary_function<real7, bool> {
    real3 _center;
    real _radius;
  public:
    is_out_of_circle_2(real3 center, real radius)
    :_center(center), _radius(radius){};
    __host__ __device__
    bool operator()(const real7 &t){
      real3 p = real3(get<0>(t), get<1>(t), get<2>(t));
      real distance = get_distance(_center, p);
      return distance > _radius;
    }
  };
}

int main(int narg, char **args){
  
  const size_t N = atoi(args[1]);
  const size_t M = atoi(args[2]);
  const real s = atof(args[3]);
  const real fin = atof(args[4]);
  char *plotfile = args[5];
  char *timefile = args[6];
  
  const real m = 1;
  thrust::constant_iterator<real> m_it(m);
 
  const size_t n_cell = (2*M) * (2*M);
  
  /*
   * mutable
   */
  size_t n_particle = N * n_cell; 

  const real rad = 1;

  cell c = make_cell(
    real3(0,0,0),
    real3(rad/M, rad/M, real(1)),
    tuple3<size_t>::type(2*M,2*M,1));

  vector<real>::type x(n_particle);
  vector<real>::type y(n_particle);
  vector<real>::type z(n_particle);
  vector<real>::type u(n_particle);
  vector<real>::type v(n_particle);
  vector<real>::type w(n_particle);
  vector<real>::type in_e(n_particle);

  vector<size_t>::type idx(n_particle);

  vector<real>::type tmp1(n_cell);
  vector<real>::type tmp2(n_cell);
  vector<real>::type tmp3(n_cell);
  vector<real>::type tmp4(n_cell);
  vector<real>::type tmp5(n_cell);
  vector<real>::type tmp6(n_cell);
  vector<real>::type tmp7(n_cell);
  vector<real>::type tmp11(n_cell);
  vector<real>::type tmp12(n_cell);
  vector<real>::type tmp13(n_cell);

  vector<size_t>::type tmp8(n_cell);
  vector<size_t>::type tmp9(n_cell);
  vector<size_t>::type tmp10(n_cell);

  /*
   * alloc random positions
   */
  for(size_t i=0; i<M*2; ++i){
    for(size_t j=0; j<M*2; ++j){
      size_t ind = i*M*2 + j;
      alloc_uniform_random(
        make_cell_at(c, i, j, 0),
        N,
        thrusting::advance(ind*N, x.begin()),
        thrusting::advance(ind*N, y.begin()),
        thrusting::advance(ind*N, z.begin()),
        i);
    }
  }

  THRUSTING_PP("n_particle before removed particles", n_particle);
  const real3 center = get_center_of(c);
  THRUSTING_PP("center of decartes cell: ", center);

  n_particle = thrusting::remove_if(
    n_particle,
    thrusting::make_zip_iterator(
      x.begin(), y.begin(), z.begin(),
      u.begin(), v.begin(), w.begin(),
      in_e.begin()),
    is_out_of_circle_2(center, rad));

  THRUSTING_PP("n_particle after removed particles", n_particle);
      
  /*
   * alloc velocity toward the center
   */
  thrusting::transform(
    n_particle,
    thrusting::make_zip_iterator(
      x.begin(),
      y.begin(),
      z.begin()),
    thrusting::make_zip_iterator(
      u.begin(),
      v.begin(),
      w.begin()),
    alloc_normal_velocity_functor(center));

  /*
   * calc initial state
   */
  thrusting::transform(
    n_particle,
    thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin()),
    idx.begin(),
    make_cellidx1_calculator(c));

  thrusting::sort_by_key(
    n_particle,
    idx.begin(),
    thrusting::make_zip_iterator(
      x.begin(), y.begin(), z.begin(),
      u.begin(), v.begin(), w.begin(),
      in_e.begin()));

  bucket_indexing(
    n_particle,
    idx.begin(),
    n_cell,
    tmp8.begin(),
    tmp9.begin());

  std::cout << "check" << std::endl;
  FILE *f = fopen("initial_noh2d_xy.dat", "w");
  for(int i=0; i<2*M; ++i){
    for(int j=0; j<2*M; ++j){
      size_t ind = i*2*M + j; 
      std::cout << i << " " << j << " " << ind << std::endl;
      assert(j < 2*M);
      real x = ((real) tmp9[ind]) / N;
      fprintf(f, "%f ", x);   
    }
    fprintf(f, "\n");    
  }
  fclose(f);

//  THRUSTING_PP("after init, x:", make_list(cnt, x.begin()));
//  THRUSTING_PP("after init, y:", make_list(cnt, y.begin()));
//  THRUSTING_PP("after init, z:", make_list(cnt, z.begin()));
//
//  THRUSTING_PP("after init, u:", make_list(cnt, u.begin()));
//  THRUSTING_PP("after init, v:", make_list(cnt, v.begin()));
//  THRUSTING_PP("after init, w:", make_list(cnt, w.begin()));

  stopwatch sw_idx("idx");
  stopwatch sw_sort_by_key("sort_by_key");
  stopwatch sw_bph("bph");
  stopwatch sw_move("move");
  stopwatch sw_boundary("boundary");

  const real dt = real(1) / M; 
  const size_t max_step =  fin / dt;

  for(size_t i=0; i<max_step; ++i){
    std::cout << "step:" << i << std::endl;
    std::cout << "time:" << dt*i << std::endl;

    sw_idx.begin();
    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin()),
      idx.begin(),
      make_cellidx1_calculator(c));
    sw_idx.end();

    sw_sort_by_key.begin();
    thrusting::sort_by_key(
      n_particle,
      idx.begin(),
      thrusting::make_zip_iterator(
        x.begin(), y.begin(), z.begin(),
        u.begin(), v.begin(), w.begin(),
        in_e.begin()));
    sw_sort_by_key.end();

    /*
     * processed by BPH routine
     */
    sw_bph.begin();
    bph(
      n_particle,
      x.begin(), y.begin(), z.begin(),
      u.begin(), v.begin(), w.begin(),
      m,
      in_e.begin(),
      s,
      idx.begin(),
      n_cell,
      // real tmp
      tmp1.begin(), tmp2.begin(), tmp3.begin(), tmp4.begin(), tmp5.begin(), 
      tmp6.begin(), tmp7.begin(), tmp11.begin(), tmp12.begin(), tmp13.begin(),
      // int tmp
      tmp8.begin(), tmp9.begin(),
      i); // seed 
    sw_bph.end();
  
    /*
     * Move
     */
    sw_move.begin();
    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin(), u.begin(), v.begin(), w.begin(), m_it), // input
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin(), u.begin(), v.begin(), w.begin()), // output 
      make_runge_kutta_1_functor(
        make_no_force_generator(),
        dt));
    sw_move.end();

    sw_boundary.begin();
    sw_boundary.end();
  } // END for

  thrusting::transform(
    n_particle,
    thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin()),
    idx.begin(),
    make_cellidx1_calculator(c));

  thrusting::sort_by_key(
    n_particle,
    idx.begin(),
    thrusting::make_zip_iterator(
      x.begin(), y.begin(), z.begin(),
      u.begin(), v.begin(), w.begin(),
      in_e.begin()));

  bucket_indexing(
    n_particle,
    idx.begin(),
    n_cell,
    tmp8.begin(),
    tmp9.begin());

  FILE *fp = fopen(plotfile, "w");
  for(int i=0; i<2*M; ++i){
    for(int j=0; j<2*M; ++j){
      size_t ind = i*2*M + j; 
      real x = ((real) tmp9[ind]) / N;
      fprintf(fp, "%f ", x);   
    }
    fprintf(fp, "\n");    
  }
  fclose(fp);

  // time data
  FILE *fp2 = fopen(timefile, "w");
  fprintf(fp2, "idx:%f\n", sw_idx.average());
  fprintf(fp2, "sort:%f\n", sw_sort_by_key.average());
  fprintf(fp2, "bph:%f\n", sw_bph.average());
  fprintf(fp2, "move:%f\n", sw_move.average());
  fprintf(fp2, "boundary:%f\n", sw_boundary.average());
  fclose(fp2);

  return 0;
}
