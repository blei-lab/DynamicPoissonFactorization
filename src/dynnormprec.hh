#ifndef DYNNORMPREC_HH
#define DYNNORMPREC_HH

#include "env.hh"
#include "normbase.hh"
#include "ratings.hh"

class DynNormPRec {
public:
  DynNormPRec(Env &env, Ratings &ratings);
  ~DynNormPRec();

  void gen_ranking_for_users(bool load_model_state);
  void load_factors(); 
  void load_correction_factors();
  void infer();
  void infer_dui();
  void infer_dui_correction();
  void elbo(); 

private:
  void initialize();
  void init_a(); 
  void approx_log_likelihood();
  void save_model(); 
  void compute_precision(bool); 
  void compute_user_precision(bool);

  void get_phi(NormBase<Matrix> &a, uint32_t ai,
            NormBase<Matrix> &b, uint32_t bi,
             Array &phi);
  void get_phi(NormBase<Matrix> &a, uint32_t ai,
            NormBase<Matrix> &b, uint32_t bi,
            NormBase<Matrix> &c, NormBase<Matrix> &d,
             Array &phi);
  void get_phi(NormBase<Matrix> &a, uint32_t ai,
	       NormBase<Matrix> &b, uint32_t bi,
	       double biasa, double biasb,
	       Array &phi);

  void load_validation_and_test_sets();
  void compute_likelihood(bool validation);
  // double log_factorial(uint32_t n)  const;

  double rating_likelihood(uint32_t p, uint32_t q, uint32_t t, yval_t y) const;
  // double rating_likelihood_hier(uint32_t p, uint32_t q, yval_t y) const;
  double prediction_score(uint32_t p, uint32_t q, uint32_t t) const;
  double score(uint32_t p, uint32_t q, uint32_t t) const;

  double remove_past_contributions(uint32_t time_curr); 

  uint32_t duration() const;
  bool is_validation(uint32_t n, uint32_t m, uint32_t ts) const;
  void explain_opt(bool optimize_correction, bool optimize_global);
  void do_on_stop(); 
    

  Env &_env;
  Ratings &_ratings;

  uint32_t _n;
  uint32_t _m;
  uint32_t _k;
  // uint32_t _t; // for time-series model 
  uint32_t _iter;
  
  vector<NormMatrix *> _thetas;
  vector<NormMatrix *> _betas;
  NormMatrix _beta, _theta;

  //Matrix a; 

  //GPArray _thetarate;
  //GPArray _betarate;
  
  CountMap _validation_map;
  CountMap _test_map;

  UserMap _sampled_users;
  UserMap _sampled_movies;

  uint32_t _start_time;
  gsl_rng *_r;

  FILE *_hf;
  FILE *_vf;
  FILE *_trf;
  FILE *_tf;
  FILE *_af;
  FILE *_pf;
  FILE *_df;

  bool _use_rate_as_score;
  uint32_t _topN_by_user;
  uint32_t _maxval, _minval;
  double _prev_h;
  uint32_t _nh;
};

inline uint32_t
DynNormPRec::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline bool
DynNormPRec::is_validation(uint32_t n, uint32_t m, uint32_t ts) const
{
  assert (n  < _n && m < _m);

  for(uint32_t t=ts; t<_env.time_periods; ++t) {
    Rating r(n,m,t);
    CountMap::const_iterator itr = _validation_map.find(r);
    if (itr != _validation_map.end()) {
      return true;
    }
  }
  return false;
}

inline void
DynNormPRec::explain_opt(bool optimize_correction, bool optimize_global)
{
    printf("\n");
    if(optimize_global)
        printf("optimize_global\n"); 
    if(optimize_correction)
        printf("optimize_correction\n"); 
}

#endif 
