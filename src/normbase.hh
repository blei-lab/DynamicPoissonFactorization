#ifndef NORMBASE_HH
#define NORMBASE_HH

#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sort_vector.h>
#include <math.h>
#include <cmath> 

#define CHECK_GRADIENTS 0

template <class T>
class NormBase { 

public:
  NormBase(string name = ""): _name(name) { }
  virtual ~NormBase() { }
  virtual const T &expected_v() const = 0;
  virtual uint32_t n() const = 0;
  virtual uint32_t k() const = 0;
  virtual void save_state(const IDMap &m) const = 0;
  string name() const { return _name; }
  double elbo() const;
private:
  string _name;
};

const uint32_t _cg_max_iter = 500; // TODO 
const float _cg_convergence  = 1e-5; // TODO

class NormMatrix : public NormBase<Matrix> {

public:
    NormMatrix(string name, double c, double d,
	   uint32_t n, uint32_t k,
	   gsl_rng **r): 
    NormBase<Matrix>(name),
    _n(n), _k(k),
    _mprior(c), // mean 
    _vprior(d), // variance
    _mcurr(n,k),
    _mnext(n,k),
    _vnext(n,k),
    _vcurr(n,k),
    _Eexpv(n,k),
    _r(r) { } 
  virtual ~NormMatrix() {} 

  void pvec(const gsl_vector *x);

  void vprint(const gsl_vector *x);
  double vnorm(const gsl_vector *x);

  uint32_t n() const { return _n;}
  uint32_t k() const { return _k;}

  const Matrix &mean_curr() const         { return _mcurr; }
  const Matrix &var_curr() const          { return _vcurr; }
  const Matrix &mean_next() const         { return _mnext; }
  const Matrix &var_next() const          { return _vnext; }
  const Matrix &expected_v() const        { return mean_curr(); }
  const Matrix &expected_expv() const  { return _Eexpv; } 

  Matrix &mean_curr()       { return _mcurr; }
  Matrix &var_curr()        { return _vcurr; }
  Matrix &mean_next()       { return _mnext; }
  Matrix &var_next()        { return _vnext; }
  Matrix &expected_v()      { return mean_curr(); }
  Matrix &expected_expv()   { return _Eexpv; } 

  const double mprior() const { return _mprior; }
  const double vprior() const { return _vprior; }
  void set_next_to_prior();
  void initialize(); 
  void initialize(Matrix& mprior, Matrix& vprior); 
  //void initialize2(double v);
  void update_mean_next(uint32_t n, const Array &phi, const double *other, const double *previous_mean, const double *next_mean, bool begin, bool end, int32_t max_iters); 
  void update_norm_mean_next(uint32_t n, const Array &phi, const double *other, const double *previous_mean, const double *next_mean, bool begin, bool end, int32_t max_iters);
  void update_var_next(uint32_t n, const double *other, bool end, int32_t max_iters); 

  //void set_mean_next(Matrix &); 
  void set_var_next(); 

  void simplex_projection(const gsl_vector* x, gsl_vector* x_proj, double z=1.0);
  void vector_normalize(gsl_vector *x);
  double vector_sum(const gsl_vector *x);
  bool is_feasible(const gsl_vector* x);

  double e() const { return -1.0;}
  void save_state(const IDMap &m) const;
  void load();
  void load(string dir);
  void load_from_pf(string dir, uint32_t K, string name);

  double elbo(bool begin, bool end, const double** previous_mean, const double** next_mean) const;
  void compute_expectations();
  void sum_rows(Array &);
  void sum_eexp_rows(Array &);

  double f_mean(const gsl_vector * p, void * params); 
  void df_mean(const gsl_vector * x, void * params, gsl_vector * g); 
  void fdf_mean(const gsl_vector * x, void * params, double * f, gsl_vector *g); 
  double f_var(const gsl_vector * p, void * params); 
  void df_var(const gsl_vector * x, void * params, gsl_vector * g); 
  void fdf_var(const gsl_vector * x, void * params, double * f, gsl_vector *g); 

  static double wrap_f_mean(const gsl_vector *x, void *params);
  static void wrap_df_mean(const gsl_vector *x, void *params, gsl_vector *g);
  static void wrap_fdf_mean(const gsl_vector *x, void *params, double *f, gsl_vector *g);
  static double wrap_f_var(const gsl_vector *x, void *params);
  static void wrap_df_var(const gsl_vector *x, void *params, gsl_vector *g);
  static void wrap_fdf_var(const gsl_vector *x, void *params, double *f, gsl_vector *g);

  //void swap(Matrix &); 
  void swap();
  void swap_mean();
  void swap_var();

private:
  uint32_t _n;
  uint32_t _k;	
  gsl_rng **_r;
  double _mprior;
  double _vprior;

  Matrix _mcurr;      // current variational mean posterior 
  Matrix _mnext;      // to help compute gradient update
  Matrix _vcurr;      // current variational variance posterior
  Matrix _vnext;      // help compute gradient update
  Matrix _Eexpv;      // expected exp weights under variational
        		      // distribution
};

typedef struct bundle {
    NormMatrix *NormMatrixObj; 
    const Array *phi;
    const double *other;
    uint32_t id; 
    const double *previous_mean, *next_mean;
    bool begin, end; 
} bundle;

inline void
NormMatrix::load()
{
  string mean_fname = name() + "_mean.tsv";
  string var_fname  = name() + "_var.tsv";
  _mcurr.load(mean_fname);
  _vcurr.load(var_fname);
  compute_expectations();

}

inline void
NormMatrix::load(string dir)
{
  string mean_fname = dir + name() + "_mean.tsv";
  string var_fname  = dir + name() + "_var.tsv";
  printf("loading %s\n", mean_fname.c_str());
  printf("loading %s\n", var_fname.c_str());
  _mcurr.load(mean_fname);
  _vcurr.load(var_fname);
  compute_expectations();

  #if 0
  double **md = _mcurr.data();
  double **vd = _vcurr.data();

  printf("md[0]0] %f\n", md[0][0]);
  for (uint32_t n=0; n< _n; ++n) {
    for (uint32_t k=0; k < _k; ++k) {
        md[n][k] += -2.3 + md[n][k] * 1e-1 *(gsl_rng_uniform(*_r) - 0.5);
        vd[n][k] += -2.3 + vd[n][k] * 1e-1 *(gsl_rng_uniform(*_r) - 0.5);
        if(vd[n][k] < 1e-5)
            vd[n][k] = 1e-5;
    }
  }
  printf("md[0]0] %f\n", md[0][0]);
  printf("vd[0][0] %f\n", vd[0][0]);
  #endif
}



inline void
NormMatrix::load_from_pf(string dir, uint32_t K, string name="")
{
  char buf_shape[1024], buf_rate[1024];

  if(name=="")
    name = this->name();

  sprintf(buf_shape, "%s/pf-fits/%s_shape-k%d.tsv", dir.c_str(), name.c_str(), K);
  lerr("loading from %s", buf_shape);
  sprintf(buf_rate, "%s/pf-fits/%s_rate-k%d.tsv", dir.c_str(), name.c_str(), K);
  lerr("loading from %s", buf_rate);

  // total hack
  //  - load Gamma shape & rate into mean and variance (that memory is already allocated)
  //  - then go through and overwrite by mean (m/v) and variance (m/v^2) 
  _mcurr.load(buf_shape, 2);
  // pf keeps a single rate per entity (item/user)
  Array rate(_n);
  rate.load(buf_rate);

  double **md = _mcurr.data();
  double **vd = _vcurr.data();
  double **ed = _Eexpv.data();
  double *rd = rate.data();

  double mean2, var;
  for (uint32_t i = 0; i < _n; ++i) {
    for (uint32_t k = 0; k < _k; ++k) {

      mean2 = pow(md[i][k]/rd[k],2);
      var   = md[i][k]/pow(rd[k],2);
      md[i][k] = log( mean2 / sqrt(mean2 + var));
      vd[i][k] = log(1 + var/mean2); 

      ed[i][k] = exp(md[i][k] + vd[i][k]/2); // exp(mean + var/2)
    

    }
  }
  //IDMap m;
  //string expv_fname = string("/") + name() + ".tsv";
  //_Ev.save(Env::file_str(expv_fname), m);
}


inline void
NormMatrix::sum_rows(Array &v)
{
    const double **ev = _mcurr.const_data();
    for (uint32_t i = 0; i < _n; ++i)
        for (uint32_t k = 0; k < _k; ++k)
            v[k] += ev[i][k];
} 

inline void
NormMatrix::sum_eexp_rows(Array &v)
{
    const double **ev = _Eexpv.const_data();
    for (uint32_t i = 0; i < _n; ++i)
        for (uint32_t k = 0; k < _k; ++k)
            v[k] += ev[i][k];
} 

inline void
NormMatrix::compute_expectations()
{ 
  // compute expectation at the point estimates of the var. distribution 
  const double ** const md = _mcurr.const_data();
  const double ** const vd = _vcurr.const_data();
  double **ed1 = _Eexpv.data();
  for (uint32_t i = 0; i < _mcurr.m(); ++i)
    for (uint32_t j = 0; j < _mcurr.n(); ++j) {
      ed1[i][j] = exp(md[i][j] + vd[i][j]/2); 
    }
}

// static wrapper-function to be able to callback the member function Display()
inline double 
NormMatrix::wrap_f_mean(const gsl_vector *x, void * params)
{
   NormMatrix& obj = (*((bundle *) params)->NormMatrixObj);
   return obj.f_mean(x, params);
}

inline void
NormMatrix::save_state(const IDMap &m) const
{
  //string expv_fname = string("/") + name() + ".tsv";
  string mean_fname = string("/") + name() + "_mean.tsv";
  string var_fname = string("/") + name() + "_var.tsv";
  string expv_fname = string("/") + name() + ".tsv";
  _mcurr.save(Env::file_str(mean_fname), m);
  _vcurr.save(Env::file_str(var_fname), m);
  _Eexpv.save(Env::file_str(expv_fname), m);
}

inline double 
NormMatrix::elbo(bool begin, bool end, const double **previous_mean=NULL, const double **next_mean=NULL) const
{ 
  double s = 0.;
  double prior_mean;

  const double **vd = _vcurr.const_data();
  const double **md = _mcurr.const_data();

  // A, B, C, D
  for(uint32_t n=0; n<_n; ++n) { 

    for(uint32_t k=0; k<_k; ++k)  {

      // sigma in derivation -> _vprior here. 
      // -log(d\sqrt(2*pi)) -d^2\nu^2
      s -= vd[n][k] / _vprior;

      if(begin) { 
        prior_mean = _mprior;
      } else { 
        prior_mean = previous_mean[n][k];
      } 
      if(!end) { 
        s -= vd[n][k] / _vprior; // A: \sigma^2 \nu^2_{t-1}
        // double counting
        //s -= std::pow(next_mean[n][k] - md[n][k], 2) / (2 * _vprior);
      }

      // - (mu - mu_t-1)^2 / (2*d^2)
      s -= std::pow(md[n][k] - prior_mean, 2)/ (2 * _vprior);

      // 1/2 * ( log \nu^2 + log 2\pi + 1)
      s += 0.5 * ( log(vd[n][k]) + log(2*M_PI) + 1);
    } // k

  } // n

  return s;
}

inline void 
NormMatrix::wrap_df_mean(const gsl_vector *x, void *params, gsl_vector *g)
{
   NormMatrix& obj = (*((bundle *) params)->NormMatrixObj);
   obj.df_mean(x, params, g);
}

inline void 
NormMatrix::wrap_fdf_mean(const gsl_vector *x, void * params, double *f, gsl_vector *g)
{
   //NormMatrix* me = (NormMatrix*) NormMatrixObj;
   NormMatrix& obj = (*((bundle *) params)->NormMatrixObj);
   obj.fdf_mean(x, params, f, g);
}

// static wrapper-function to be able to callback the member function Display()
inline double 
NormMatrix::wrap_f_var(const gsl_vector *x, void * params)
{
   NormMatrix& obj = (*((bundle *) params)->NormMatrixObj);
   return obj.f_var(x, params);
}

inline void 
NormMatrix::wrap_df_var(const gsl_vector *x, void *params, gsl_vector *g)
{
   NormMatrix& obj = (*((bundle *) params)->NormMatrixObj);
   obj.df_var(x, params, g);
}

inline void 
NormMatrix::wrap_fdf_var(const gsl_vector *x, void * params, double *f, gsl_vector *g)
{
   NormMatrix& obj = (*((bundle *) params)->NormMatrixObj);
   obj.fdf_var(x, params, f, g);
}


inline double 
NormMatrix::f_mean(const gsl_vector *x, void * params)
{
    bundle &b = (bundle &) (*(bundle *) (params)); 
    NormMatrix& obj = (*b.NormMatrixObj);
    uint32_t id = b.id;

    // No time
    //f = (x_k - c)^2 / 2*d^2 - exp(x + nu^2/2) + \phi_mn * x

    // W. time (and t > 1. when t=1, replace x_k,t-1 by c)
    // f = - (x_k - x_k,t-1)^2 / 2*d^2*delta 
    //     - (x_k,t+1 - x_k)^2 / 2*d^2*delta  - exp(x + nu^2/2) + \phi_mn * x 

    gsl_vector *tmp1 = gsl_vector_calloc(obj._k);
    gsl_vector *tmp2 = gsl_vector_calloc(obj._k);

    // (x_k - x_k,t-1)^2 / 2*d^2
    for(uint32_t k=0;k<obj._k;++k) { 
      if (b.begin)
          gsl_vector_set(tmp1, k, gsl_vector_get(x, k) - b.previous_mean[k]); 
      else 
          gsl_vector_set(tmp1, k, gsl_vector_get(x, k) - b.previous_mean[k]); 
    }
    gsl_vector_mul(tmp1, tmp1);
    gsl_vector_scale(tmp1, 1/(2*obj._vprior));

    //gsl_vector *xa = gsl_vector_calloc(obj._k);
    // (x_k,t+1 - x_k)^2 / 2*d^2*delta
    if (!b.end) { 
      for(uint32_t k=0;k<obj._k; ++k) {
        //gsl_vector_set(xa, k, gsl_vector_get(x, k));
        gsl_vector_set(tmp2, k, b.next_mean[k] - gsl_vector_get(x, k));
      }
      //gsl_vector_sub(tmp2, xa);
      gsl_vector_mul(tmp2, tmp2);
      gsl_vector_scale(tmp2, 1/(2*(obj._vprior)));
    } 

    const double ** const vd = obj._vcurr.const_data();
    const double * const pd = b.phi->const_data(); 
    const double * const od = b.other; 
    double x_k; 

    double f = .0; 
    for(uint32_t k=0; k<obj._k; ++k) { 

        f -= gsl_vector_get(tmp1, k); 
        f -= gsl_vector_get(tmp2, k); 
        x_k = gsl_vector_get(x, k);
        f -= exp(x_k + vd[id][k]/2) * od[k];
        f += pd[k] * x_k;
        if (std::isnan(f)) {
            printf("f is nan (tmp1k: %f, tmp2k: %f, x_k %f, vd %f, od %f, pd %f)\n", 
                    gsl_vector_get(tmp1, k), gsl_vector_get(tmp2, k), 
                    x_k, vd[id][k], od[k], pd[k]); 
            exit(-1);
        }
    }

    gsl_vector_free(tmp1); 
    gsl_vector_free(tmp2); 

    return -f; // minimize
}


inline void 
NormMatrix::df_mean(const gsl_vector * x, void * params, gsl_vector * df)
{
    bundle &b = (bundle &) (*(bundle *) (params)); 
    NormMatrix& obj = (*b.NormMatrixObj);
    uint32_t id = b.id;

    // W. time 
    // f = - (x_k - x_k,t-1) / d^2*delta 
    //     + (x_k,t+1 - x_k) / d^2*delta  - exp(x + nu^2/2) + \phi_mn

    for(uint32_t k=0; k<obj._k; ++k) {
      if (b.begin)
        gsl_vector_set(df, k, gsl_vector_get(x,k) - b.previous_mean[k]); 
      else 
        gsl_vector_set(df, k, gsl_vector_get(x,k) - b.previous_mean[k]);
    }
    gsl_vector_scale(df, -1/obj._vprior);

    if (!b.end) { 
      gsl_vector *tmp2 = gsl_vector_alloc(obj._k); 
      for(uint32_t k=0; k<obj._k; ++k)
        gsl_vector_set(tmp2, k, b.next_mean[k] - gsl_vector_get(x,k)); 
        // gsl_vector_set(tmp2, k, b.next_mean[k] - gsl_vector_get(x,k)); 
      gsl_vector_scale(tmp2, 1/(obj._vprior));
      gsl_vector_add(df, tmp2); 
      gsl_vector_free(tmp2); 
    }

    // probably FASTER way of doing it
    // (time > 1)
    // f = x_k,t-1 - x_k,t+1 / d^2*delta - exp(x + nu^2/2) + \phi_mn
    // (time == beg.)
    // - (x_k,t+1 - x_k) / d^2*delta - exp(x + nu^2/2) + \phi_mn
    // (time == end)
    // - (x_k - x_k,t-1) / d^2*delta - exp(x + nu^2/2) + \phi_mn - exp(x + nu^2/2) + \phi_mn


    // (x_k / d^2) 
    //gsl_vector_memcpy(df, x);
    //gsl_vector_scale(df, 1/(obj._vprior*obj._vprior));

    const double ** const vd = obj._vcurr.const_data();
    const double * const pd = b.phi->const_data(); 
    const double * const od = b.other; 

    //gsl_vector_add_constant(df, obj._mprior/(obj._vprior*obj._vprior));

    double x_k, df_k; 
    for(uint32_t k=0; k<obj._k; ++k) { 
        x_k  = gsl_vector_get(x, k); 
        df_k = gsl_vector_get(df, k);
        gsl_vector_set(df, k, df_k -exp(x_k + vd[id][k]/2)*od[k] + pd[k]);
    }
    gsl_vector_scale(df, -1); // maximize

}

inline void 
NormMatrix::fdf_mean(const gsl_vector * x, void * params, double * f, gsl_vector * df)
{
    *f = f_mean(x, params);
    df_mean(x, params, df);
}



inline void
NormMatrix::set_next_to_prior()
{
  _mnext.set_elements(0.);
  _vnext.set_elements(0.);
  //_mnext.set_elements(_mprior);
  //_vnext.set_elements(_vprior);
}

inline void
NormMatrix::swap()
{
  _mcurr.swap(_mnext);
  _vcurr.swap(_vnext);
  set_next_to_prior();
}

inline void
NormMatrix::swap_mean()
{
  _mcurr.swap(_mnext);
  _mnext.set_elements(0.);
}


inline void
NormMatrix::swap_var()
{
  _vcurr.swap(_vnext);
  _vnext.set_elements(0.);
}

/*
inline void
NormMatrix::swap(Matrix &m)
{
  _mcurr.swap(_mnext);
  _vcurr.swap(_vnext);
  set_mean_next(m);
  set_var_next(); 
}
*/

/*
inline void 
NormMatrix::set_mean_next(Matrix &m)
{
  double **mnd = _mnext.data(); 
  const double **md = m.const_data();

  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      mnd[i][k] = 0.; 
      //mnd[i][k] = md[i][k];
}
*/

inline void 
NormMatrix::set_var_next()
{
  //_vnext.set_elements(_vprior);
  _vnext.set_elements(0.);
}


/*
inline void
NormMatrix::initialize(Matrix& mprior, Matrix& vprior)
{
  double **md = _mcurr.data();
  double **vd = _vcurr.data();

  const double **mpd = mprior.const_data();
  const double **vpd = vprior.const_data();

  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      md[i][k] = mpd[i][k] + 0.01 * gsl_rng_uniform(*_r);
      //gsl_ran_gaussian_ziggurat(*_r, _vprior)

  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      vd[i][k] = vpd[i][k] + 0.01 * gsl_rng_uniform(*_r);

  set_mean_next(mprior); 
  set_var_next(); 
}
*/


inline void
NormMatrix::initialize()
{
  double **ad = _mcurr.data();
  double **bd = _vcurr.data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      //ad[i][k] = _mprior + 0.01 * gsl_rng_uniform(*_r);
      ad[i][k] = _mprior + gsl_rng_uniform(*_r) * 1e-3;
      //gsl_ran_gaussian_ziggurat(*_r, _vprior)

  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      //bd[i][k] = 0.01 * gsl_rng_uniform(*_r);
      bd[i][k] = 0.1 * gsl_rng_uniform(*_r) * 1e-3;
      //bd[i][k] = _vprior + 0.01 * gsl_rng_uniform(*_r);

  set_next_to_prior();
}

#if 0
inline void
NormMatrix::initialize2(double v)
{
  double **ad = _mcurr.data();
  double **bd = _vcurr.data();
  for (uint32_t i = 0; i < _n; ++i) {
    for (uint32_t k = 0; k < _k; ++k) {
      ad[i][k] = _mprior + 0.01 * gsl_rng_uniform(*_r);
      bd[i][k] = _vprior + 0.01 * gsl_rng_uniform(*_r);
    }
  }
  set_next_to_prior();
}
#endif

inline void 
NormMatrix::pvec(const gsl_vector *x) { 
    double a;
    bool nan=false;
    for (uint32_t k=0; k<x->size; ++k) {
        a = gsl_vector_get(x, k);
        printf("%f ", a);
        if (std::isnan(a))
            nan=true;
    }
    printf("\n"); 
    
    if (nan)
        exit(-1);
}

inline double
NormMatrix::vnorm(const gsl_vector *x) {
  double n=0.;
  for (uint32_t k=0; k<x->size; ++k)
    n += gsl_vector_get(x,k) * gsl_vector_get(x,k);
  return sqrt(n);
}

inline void
NormMatrix::vprint(const gsl_vector *x) {
    for (uint32_t k=0; k<x->size; ++k)
        printf("%f ", gsl_vector_get(x, k));
    printf("\n");
}

inline double 
NormMatrix::f_var(const gsl_vector *x, void * params)
{
    bundle &b = (bundle &) (*(bundle *) (params));
    NormMatrix& obj = (*b.NormMatrixObj); 
    uint32_t id = ((bundle *) params)->id;
    const double ** const md = obj._mcurr.const_data();
    const double * const od = b.other;

    double f = .0; 

    //f = - d^2 nu_k^2 + 1/2(log \nu^2) - exp(x + nu^2/2)
    double exp_x_k; 
    for(uint32_t k=0; k<obj._k; ++k) { 

        exp_x_k = exp(gsl_vector_get(x, k));

        // -d^2 \nu_k^2 + 1/2(log \nu^2) - exp(x + nu^2/2)
        f -= (b.end==true?1:2)*exp_x_k/obj._vprior;
        f += 0.5 * gsl_vector_get(x,k);
        f -= exp(md[id][k] + exp_x_k/2) * od[k];

        if(std::isnan(f)) { 
            printf("nan (%d). exp_x_k: %f; x_k: %f; obj._vprior %f; od[k]: %f; f: %f; md: %f \n", k, exp_x_k, log(exp_x_k), obj._vprior, od[k], f, md[id][k]);
            exit(-1);
        }
    }

    return -f; // maximize
}


inline void 
NormMatrix::df_var(const gsl_vector * x, void * params, gsl_vector * df)
{
    bundle &b = (bundle &) (*(bundle *) (params));
    NormMatrix& obj = (*b.NormMatrixObj);
    uint32_t id = ((bundle *) params)->id;

   //df = -d^2 + 1/(2nu_k^2) - 1/2exp(x + nu^2/2) 

    const double ** const md = obj._mcurr.const_data();
    const double * const od = b.other; 

    double exp_x_k; 
    for(uint32_t k=0; k<obj._k; ++k) { 
        exp_x_k = exp(gsl_vector_get(x, k)); 
        //gsl_vector_set(df, k, -obj._vprior*obj._vprior + 0.5/x_k - 0.5*exp(md[id][k] + x_k/2));
        gsl_vector_set(df, k, -(b.end == true?1:2)*exp_x_k/obj._vprior
                              + 0.5
                              - 0.5*exp_x_k*exp(md[id][k] + exp_x_k/2)*od[k]);
    }

    gsl_vector_scale(df, -1.0); // maximize
}

inline void 
NormMatrix::fdf_var(const gsl_vector * x, void * params, double * f, gsl_vector * df)
{
    *f = f_var(x, params);
    df_var(x, params, df);
}


inline void
NormMatrix::update_mean_next(uint32_t n, const Array &phi, const double *other, const double *previous_mean, const double *next_mean, bool begin, bool end, int32_t max_iters=-1)
{

    //printf("optimizing user %d\n", n); 
    max_iters = -1; 
    // update the K-vector mean of a particular user/item
    // involves 
    // 1) phi 
    // 2) priors (c,d)
    // 3) nu
    // 4) previous mean  (NEW) -> that should take the place of c. 
    // 5) next mean      (NEW) -> needs the addition of a new term (which
    // is similar to the previous term involving c)
    // keep them all in a bundle 

    gsl_multimin_fdfminimizer * s;
    gsl_multimin_function_fdf mu_obj;

    uint32_t iter = 0;
    int status;
    double f_old, converged;

    bundle b;
    b.NormMatrixObj = this; 
    b.phi = &phi;
    b.other = other; 
    b.id = n; 
    b.previous_mean = previous_mean; 
    b.next_mean     = next_mean; 
    b.begin         = begin; 
    b.end           = end; 

    mu_obj.f = &NormMatrix::wrap_f_mean;
    mu_obj.df = &NormMatrix::wrap_df_mean;
    mu_obj.fdf = &NormMatrix::wrap_fdf_mean;
    mu_obj.n = _k;
    mu_obj.params = (void *)&b;

    // starting value
    const gsl_multimin_fdfminimizer_type * T;
    // T = gsl_multimin_fdfminimizer_vector_bfgs;
    // T = gsl_multimin_fdfminimizer_conjugate_fr;
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, _k);

    gsl_vector* x = gsl_vector_calloc(_k);
    const double ** md = _mcurr.const_data();
    for (uint32_t k = 0; k < _k; ++k) gsl_vector_set(x, k, md[n][k]);

    #if CHECK_GRADIENTS
    gsl_vector *xg = gsl_vector_alloc(_k);
    gsl_vector *dfh  = gsl_vector_alloc(_k);
    double eps = 1e-5;
    double f1, f2;
    for(uint32_t k=0; k < _k; ++k) {
      gsl_vector_memcpy(xg, x);
      gsl_vector_set(xg, k, gsl_vector_get(xg, k) + eps);
      f1 = wrap_f_mean(xg, (void *)&b);
      gsl_vector_set(xg, k, gsl_vector_get(xg, k) - 2*eps);
      f2 = wrap_f_mean(xg, (void *)&b);
      gsl_vector_set(dfh, k, (f1-f2)/(2*eps));
    }
    gsl_vector* df = gsl_vector_calloc(_k);
    wrap_df_mean(x, (void *)&b, df);
    gsl_vector *t1 = gsl_vector_calloc(_k);
    gsl_vector *t2 = gsl_vector_calloc(_k);
    gsl_vector_memcpy(t1,dfh);
    gsl_vector_sub(t1,df);
    gsl_vector_memcpy(t2,dfh);
    gsl_vector_add(t2,df);
   
    //norm(dh-dy)/norm(dh+dy);
    if(vnorm(t1)/vnorm(t2) > 1e-5) {
        vprint(x);
        printf("mean gradient doesn't match %e", vnorm(t1)/vnorm(t2));
        for(uint32_t k=0; k<_k;++k)
          printf(" (%.10f,%.10f)", gsl_vector_get(df,k), gsl_vector_get(dfh,k));
        printf("n: %d\n", n); 
        printf("\n");
        exit(-1); 
    } // else { printf("\n\nmean grad matches\n"); }
    gsl_vector_free(t1);
    gsl_vector_free(t2);
    gsl_vector_free(xg);
    gsl_vector_free(dfh);
    gsl_vector_free(df);

    #endif

    //printf("x before optimization:\n"); 
    gsl_multimin_fdfminimizer_set(s, &mu_obj, x, 0.01, 1e-3);

    do
    {
        iter++;
        f_old = s->f;
        status = gsl_multimin_fdfminimizer_iterate(s);
        converged = fabs((f_old - s->f) / f_old);
        //printf("f(mu) = %5.17e ; conv = %5.17e\n", s->f, converged);
        //pvec(s->x); 
        if (status) break;
        status = gsl_multimin_test_gradient(s->gradient, _cg_convergence);
    }
    while ((status == GSL_CONTINUE) && (iter < _cg_max_iter) && (iter < max_iters || max_iters == -1));
    // while ((converged > PARAMS.cg_convergence) &&
    // ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    #if 0
    if (iter == _cg_max_iter) {
        printf("\nwarning: cg didn't converge (mu) %d\n", _cg_max_iter);
        printf("x %d: (%d)\n", n, _k);
        pvec(s->x); 
        printf("nu^2\n"); 
        const double ** const vd = _vcurr.const_data();
        for (uint32_t k=0; k<_k; ++k)
            printf("%f ", vd[n][k]); 
        printf("\n"); 
        
        const double * const pd = phi.const_data(); 
        printf("phi\n"); 
        for (uint32_t k=0; k<_k; ++k)
            printf("%f ", pd[k]); 
        printf("\n"); 
    }
    #endif
    #if 0
    if(iter < 2 && status != GSL_SUCCESS && status != GSL_CONTINUE) {
        printf("failed mean opt. iter %d-%d\n", iter, status);
        pvec(s->x); 
        printf("other\n");
        for(int k=0; k<_k; ++k)
            printf("%f ", other[k]); 
        printf("\n"); 
        exit(-1); 
    }
    #endif

    //printf("x mean after optimization:\n"); 
    //pvec(s->x, _k); 

    // set result 
    Array mean(_k); 
    for (uint32_t k = 0; k < _k; ++k)
        mean[k] = gsl_vector_get(s->x, k);
    _mnext.set_slice(n, mean);
    _mcurr.set_slice(n, mean);

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);

}

// projected gradient code mostly from Chong Wang's CTR code (http://www.cs.princeton.edu/~chongw/software/ctr.tar.gz)
inline void
NormMatrix::update_norm_mean_next(uint32_t n, const Array &phi, const double *other, const double *previous_mean, const double *next_mean, bool begin, bool end, int32_t max_iters=-1)
{

    // Take a single gradient step and project to the simplex. then return. 
    // We could repeat 

#if 0 

    //for (int inter=0; iter<1; ++iter) { 
  
        df_mean(const gsl_vector * x, void * params, gsl_vector * df)

        // Project to the simplex  

        // unormalized version: s->x
        gsl_vector * x_proj = gsl_vector_alloc(s->x->size); 
        norm_x_simplex(s->x, x_proj, x_old); 

        simplex_projection(x, x_proj); 

    //}

    // set result 
    Array mean(_k); 
    for (uint32_t k = 0; k < _k; ++k)
        mean[k] = gsl_vector_get(s->x, k);
    _mnext.add_slice(n, mean);
#endif
////////
  bundle b;
  b.NormMatrixObj = this; 
  b.phi = &phi;
  b.other = other; 
  b.id = n; 
  b.previous_mean = previous_mean; 
  b.next_mean     = next_mean; 
  b.begin         = begin; 
  b.end           = end; 

  gsl_vector* gradient = gsl_vector_alloc(_k);
  gsl_vector* x_proj = gsl_vector_alloc(_k);
  gsl_vector* x = gsl_vector_alloc(_k);
  gsl_vector* x_old = gsl_vector_alloc(_k);
  const double ** d = _mcurr.const_data(); 
  for (uint32_t i=0; i<_mcurr.n(); ++i) {
    gsl_vector_set(x_old, i, d[n][i]); 
    gsl_vector_set(x, i, d[n][i]); 
  }

  double f_old = f_mean(x, (void *)&b);  // function is meant to minimize
  //printf("f_old: %0.10f -> ", f_old); 

  df_mean(x, (void *)&b, gradient);
  double ab_sum = gsl_blas_dasum(gradient); // absolute sum
  if (ab_sum > 1.0) gsl_vector_scale(gradient, 1.0/ab_sum); // rescale the gradient
  
  gsl_blas_daxpy(-1, gradient, x); // apply gradient
  simplex_projection(x, x_proj);
  gsl_vector_sub(x_proj, x_old); // x_proj is still on the simplex
  double r = 0; 
  gsl_blas_ddot(gradient, x_proj, &r);
  
  r *= 0.5;
  
  double beta = 0.5;
  double f_new;
  double t = beta;
  int iter = 0;
  // through the projection the likelihood may have decreased. 
  // here we search on the line from old_x to x_proj where the likelihood is increased
  while(++iter < 100) {
    gsl_vector_memcpy(x, x_old);
    gsl_blas_daxpy(t, x_proj, x);

    f_new = f_mean(x, (void *)&b); 
    if (f_new > f_old + r * t) t = t * beta;
    else break;
  }

  if (!is_feasible(x)) { printf("sth is wrong, not feasible. you've got to check it ...\n"); exit(-1); }

  Array mean(_k); 
  for (uint32_t k = 0; k < _k; ++k) {
    //mean[k] = gsl_vector_get(x, k);
    _mnext.set(n, k, gsl_vector_get(x, k)); 
  }
  //_mnext.add_slice(n, mean);

  gsl_vector_free(gradient);
  gsl_vector_free(x);
  gsl_vector_free(x_proj);
  gsl_vector_free(x_old); 

}

inline bool 
NormMatrix::is_feasible(const gsl_vector* x) {
 double val;
 double sum  = 0;
 for (size_t i = 0; i < x->size-1; i ++) {
   val = gsl_vector_get(x, i);
   if (val < 0 || val >1) return false;
   sum += val;
   if (sum > 1) return false;
 }
 return true;
}

// project x on to simplex (using // http://www.cs.berkeley.edu/~jduchi/projects/DuchiShSiCh08.pdf)
inline void
NormMatrix::simplex_projection(const gsl_vector* x, gsl_vector* x_proj, double z) {
  gsl_vector_memcpy(x_proj, x);
  gsl_sort_vector(x_proj);
  double cumsum = -z, u;
  int j = 0;
  int i; // this has to be int, not size_t
  for (i = (int)x->size-1; i >= 0; i --) {
    u = gsl_vector_get(x_proj, i);
    cumsum += u;
    if (u > cumsum/(j+1)) j++;
    else { cumsum -= u; break; }
  }
  double theta = cumsum/j;
  for (i = 0; i < (int)x->size; i ++) {
    u = gsl_vector_get(x, i)-theta;
    if (u <= 0) u = 0.0;
    gsl_vector_set(x_proj, i, u);
  }
  vector_normalize(x_proj); // fix the normaliztion issue due to numerical errors
}

inline void 
NormMatrix::vector_normalize(gsl_vector *x)
{
  double v = vector_sum(x);
  if (v > 0 || v < 0)
    gsl_vector_scale(x, 1/v);
}

inline double 
NormMatrix::vector_sum(const gsl_vector *x)
{
  double sum=0.; 
  for (int i=0; i<x->size; ++i)
    sum += gsl_vector_get(x,i); 
  return sum;
}

inline void
NormMatrix::update_var_next(uint32_t n, const double *other, bool end=true, int32_t max_iters=-1)
{
    max_iters=-1;

    // update the K-vector mean of a particular user/item
    // involves 
    // 2) priors (c,d)
    // 3) nu
    // keep pointers to all in a bundle 

    gsl_multimin_fdfminimizer * s;
    gsl_multimin_function_fdf nu_obj;

    uint32_t iter = 0;
    int status;
    double f_old, converged;

    bundle b;
    b.NormMatrixObj = this; 
    b.id = n; 
    b.end = end; 
    b.other = other; 

    nu_obj.f = &NormMatrix::wrap_f_var;
    nu_obj.df = &NormMatrix::wrap_df_var;
    nu_obj.fdf = &NormMatrix::wrap_fdf_var;
    nu_obj.n = _k;
    nu_obj.params = (void *)&b;

    // starting value
    const gsl_multimin_fdfminimizer_type * T;
    // T = gsl_multimin_fdfminimizer_vector_bfgs;
    //T = gsl_multimin_fdfminimizer_conjugate_fr;
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, this->_k);

    gsl_vector* x = gsl_vector_calloc(_k);
    double **vd = _vcurr.data();
    for (uint32_t k = 0; k < _k; ++k) gsl_vector_set(x, k, log(vd[n][k]));


    gsl_multimin_fdfminimizer_set(s, &nu_obj, x, 0.01, 1e-3);

    #if CHECK_GRADIENTS
    gsl_vector *xg = gsl_vector_alloc(_k);
    gsl_vector *dfh  = gsl_vector_alloc(_k);
    double eps = 1e-9;
    double f1, f2;
    for(uint32_t k=0; k < _k; ++k) {
      gsl_vector_memcpy(xg, x);
      gsl_vector_set(xg, k, gsl_vector_get(xg, k) + eps/2);
      f1 = wrap_f_var(xg, (void *)&b);
      gsl_vector_set(xg, k, gsl_vector_get(xg, k) - eps);
      f2 = wrap_f_var(xg, (void *)&b);
      gsl_vector_set(dfh, k, (f1-f2)/eps);
    }
    gsl_vector* df = gsl_vector_calloc(_k);
    wrap_df_var(x, (void *)&b, df);
    gsl_vector *t1 = gsl_vector_calloc(_k);
    gsl_vector *t2 = gsl_vector_calloc(_k);
    gsl_vector_memcpy(t1,dfh);
    gsl_vector_sub(t1,df);
    gsl_vector_memcpy(t2,dfh);
    gsl_vector_add(t2,df);
    
    //norm(dh-dy)/norm(dh+dy);
    if(vnorm(t1)/vnorm(t2) > 1e-4) {
        vprint(x);
        printf("var gradient doesn't match %e", vnorm(t1)/vnorm(t2)); 
        for(uint32_t k=0; k<_k;++k) 
          printf(" (%.10f,%.10f)", gsl_vector_get(df,k), gsl_vector_get(dfh,k));
        printf("\n"); 
        exit(-1);
    } //else { printf("\n\nvar grad matches\n"); }
    gsl_vector_free(t1); 
    gsl_vector_free(t2); 
    gsl_vector_free(xg);
    gsl_vector_free(dfh);
    gsl_vector_free(df);
    #endif

    do
    {
        iter++;
        f_old = s->f;
        status = gsl_multimin_fdfminimizer_iterate(s);
        converged = fabs((f_old - s->f) / f_old);
        //printf("f(nu) = %5.17e ; conv = %5.17e\n", s->f, converged);
        //pvec(s->x, _k); 
        if (status) break;
        status = gsl_multimin_test_gradient(s->gradient, _cg_convergence);
    }
    while ((status == GSL_CONTINUE) && (iter < _cg_max_iter) && (iter < max_iters || max_iters == -1));
    // while ((converged > PARAMS.cg_convergence) &&
    // ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    #if 0
    if (iter == _cg_max_iter) { 
        printf("warning: cg didn't converge (nu) %d\n", _cg_max_iter);
        printf("x\n");
        pvec(s->x); 
        printf("mu\n"); 
        const double ** const md = _mcurr.const_data();
        for (uint32_t k=0; k<_k; ++k)
            printf("%f ", md[n][k]); 
        printf("\n"); 
        
        printf("\n"); 
        exit(-1); 
    } 
    #endif

    //if (iter < 2 && status != GSL_SUCCESS && status != GSL_CONTINUE) { 
    //    printf("iter %d-%d\n", iter, status);
    //}

    //printf("x after optimization:\n"); 
    //pvec(s->x); 

    Array var(_k); 
    for (uint32_t k = 0; k < _k; ++k)
        var[k] = exp(gsl_vector_get(s->x, k));
    _vnext.set_slice(n, var);
    _vcurr.set_slice(n, var);

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);

}


#endif 

