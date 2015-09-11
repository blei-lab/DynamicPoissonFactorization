#include "dynnormprec.hh"
#include "utils.hh" 
#include <float.h>


DynNormPRec::DynNormPRec(Env &env, Ratings &ratings)
  : _env(env), _ratings(ratings),
    _n(env.n), _m(env.m), _k(env.k),
    _iter(0),
    _start_time(time(0)),
    _beta("beta", 1e-2, _env.vprior, _m,_k,&_r),
    _theta("theta", 1e-2, _env.vprior, _n,_k,&_r),
    _use_rate_as_score(true),
    _topN_by_user(100),
    _maxval(0), _minval(65536),
    _prev_h(.0), _nh(.0)
    {
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (_env.seed)
    gsl_rng_set(_r, _env.seed);
  Env::plog("infer n:", _n);

  _hf = fopen(Env::file_str("/heldout.txt").c_str(), "w");
  if (!_hf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _vf = fopen(Env::file_str("/validation.txt").c_str(), "w");
  if (!_vf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _trf = fopen(Env::file_str("/train.txt").c_str(), "w");
  if (!_trf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _tf = fopen(Env::file_str("/test.txt").c_str(), "w");
  if (!_tf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _af = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_af)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  _pf = fopen(Env::file_str("/precision.txt").c_str(), "w");
  if (!_pf)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }  
  _df = fopen(Env::file_str("/ndcg.txt").c_str(), "w");
  if (!_df)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }  

  load_validation_and_test_sets();

  // allocate user and item weights for each timestep 
  float vprior = _env.vprior;
  float mprior = 1e-2;
  for(uint32_t t=0; t <= _env.max_train_time_period; ++t)
  {
    NormMatrix *a = new NormMatrix("theta_" + std::to_string((long long int)t), mprior, vprior, _n,_k,&_r);
    _thetas.push_back(a);

    a = new NormMatrix("beta_" + std::to_string((long long int)t), mprior, vprior, _m, _k, &_r);
    _betas.push_back(a);
  } 

  Env::plog("theta mean:", _thetas[0]->mprior()); // assumes the same priors 
  Env::plog("theta var:", _thetas[0]->vprior());  // across all thetas
  Env::plog("beta mean:", _betas[0]->mprior());
  Env::plog("beta var:", _betas[0]->vprior());
}

DynNormPRec::~DynNormPRec()
{
  fclose(_hf);
  fclose(_vf);
  fclose(_trf);
  fclose(_af);
  fclose(_pf);
  fclose(_tf);
}

double old_elbo=-DBL_MAX; 

void 
DynNormPRec::elbo()
{

  uint32_t x;
  x = _k;

  double s = .0;
  bool begin, end; 

  const double **etheta = _theta.expected_v().const_data();
  const double **ebeta = _beta.expected_v().const_data();
  const double **eexptheta = _theta.expected_expv().const_data();
  const double **eexpbeta = _beta.expected_expv().const_data();
  const double **eexpthetas = NULL, **ethetas = NULL;
  const double **eexpbetas = NULL, **ebetas = NULL;

  double train_ll = 0.;
  uint32_t train_ll_k = 0;

  double v=0.;
  for (uint32_t t = 0; t <= _env.max_train_time_period; ++t) {

    ebetas = _betas[t]->expected_v().const_data();
    ethetas = _thetas[t]->expected_v().const_data();

    eexpbetas = _betas[t]->expected_expv().const_data();
    eexpthetas = _thetas[t]->expected_expv().const_data();
    
    #pragma omp parallel for num_threads(_env.num_threads)
    for (uint32_t n = 0; n < _n; ++n) {

      Array phi(x);
      double siter = 0., train_ll_iter = 0.;
      uint32_t train_ll_k_iter = 0;

      const vector<uint32_t> *movies = _ratings.get_movies(n,t);
      for (uint32_t j = 0; j < movies->size(); ++j) {
        uint32_t m = (*movies)[j];
        yval_t y = _ratings.r(n,m,t);

        get_phi(*_thetas[t], n, *_betas[t], m, _theta, _beta, phi);

        //siter -= log_factorial(y);  // shouldn't that be for all m,n,t triplets?

        for (uint32_t k = 0; k < _k; ++k) {
          siter += y * phi[k] * (ethetas[n][k] + ebetas[m][k] + etheta[n][k] + ebeta[m][k] - log(phi[k]));
        }

        // train ll 
        train_ll_iter += rating_likelihood(n,m,t,y);
        train_ll_k_iter++;

        // add fake 0  
        //train_ll_iter += rating_likelihood(n, gsl_rng_uniform_int(_r, _m), t, 0);
        //train_ll_k_iter++;

      } // items

#if 0
      if (!_env.update_a) {
        for (uint32_t m=0; m < _m; ++m)
          for (uint32_t k = 0; k < _k; ++k)
              v += eexptheta[n][k] * eexpthetas[n][k] * eexpbeta[m][k] * eexpbetas[m][k];
      }
#endif

      #pragma omp critical
      {
        s += siter;
        train_ll += train_ll_iter;
        train_ll_k += train_ll_k_iter;
      }
    } // users

      Array tmp(_k); tmp.zero();
      double *tmpd = tmp.data();
      for(uint32_t m=0; m < _m; ++m)  {
        for(uint32_t k=0; k < _k; ++k) {
          tmpd[k] += eexpbeta[m][k] * eexpbetas[m][k];
        }
      }

      for(uint32_t n=0; n < _n; ++n)  {
        for(uint32_t k=0; k < _k; ++k) {
          v += eexptheta[n][k] * eexpthetas[n][k] * tmpd[k];
        }
      }

    begin = (t==0 ? true: false); 
    end = (t==_env.max_train_time_period ? true: false); 
    if (begin) {
      s += (*_thetas[t]).elbo(begin, end, NULL, _thetas[t+1]->mean_curr().const_data());
    } else if (end) {
      s += (*_thetas[t]).elbo(begin, end, _thetas[t-1]->mean_curr().const_data());
    } else // middle 
      s += (*_thetas[t]).elbo(begin, end, _thetas[t-1]->mean_curr().const_data(), _thetas[t+1]->mean_curr().const_data());

    if (begin) {
      s += (*_betas[t]).elbo(begin, end, NULL, _betas[t+1]->mean_curr().const_data());
    } else if (end) {
      s += (*_betas[t]).elbo(begin, end, _betas[t-1]->mean_curr().const_data());
    } else // middle 
      s += (*_betas[t]).elbo(begin, end, _betas[t-1]->mean_curr().const_data(), _betas[t+1]->mean_curr().const_data());

  } // time 

  //printf("velbo %f\n", v);
  s -= v;

  Matrix onesn(_n, _k, false);
  onesn.set_elements(1.0);
  Matrix onesm(_m, _k, false);
  onesm.set_elements(1.0);
  s += _theta.elbo(true, true, onesn.const_data());
  s += _beta.elbo(true, true, onesm.const_data());

  printf("\n\t\t\tELBO: %e (train ll: %f)", s, train_ll/train_ll_k); 
  if((s - old_elbo) < -1e-5)
    printf("\n\t\t\t\t\t ELBO goes down (%e)\n", s-old_elbo);
  old_elbo=s;

  fprintf(_trf, "%.9f\t%d\n", train_ll/train_ll_k, train_ll_k);
  fflush(_trf);

  fprintf(_af, "%.5f\n", s);
  fflush(_af);

}

void
DynNormPRec::load_validation_and_test_sets()
{
  char buf[4096];
  sprintf(buf, "%s/validation.tsv", _env.datfname.c_str());
  FILE *validf = fopen(buf, "r");
  assert(validf);
  _ratings.read_generic(validf, &_validation_map);
  fclose(validf);

  sprintf(buf, "%s/test.tsv", _env.datfname.c_str());
  FILE *testf = fopen(buf, "r");
  assert(testf);
  _ratings.read_generic(testf, &_test_map);
  fclose(testf);
  
  printf("+ loaded validation and test sets from %s\n", _env.datfname.c_str());
  fflush(stdout);
  Env::plog("test ratings", _test_map.size());
  Env::plog("validation ratings", _validation_map.size());
}

void
DynNormPRec::initialize()
{
  if (_env.pf_init) {
    printf("+ loading beta/thetas from pf\n"); 
    // wasteful... 
    for(uint32_t t=0; t<_thetas.size(); ++t) { 
      _thetas[t]->load_from_pf(_env.datfname, _k, "theta"); // sets curr mean/var to pf fit
      //_thetas[t]->set_to_prior(); // sets next mean/var to prior
      //if(t==0)
      //  _thetas[t]->set_next_to_prior(); 
      //else {
      //  _thetas[t]->set_mean_next(_thetas[t-1]->mean_curr());
      //  _thetas[t]->set_var_next(); 
      //}
      _thetas[t]->compute_expectations(); 
      if(_env.normalized_representations)
        _thetas[t]->mean_curr().normalize1(); 
    }

    printf("beta dynamic loading not implemented");
    assert(0==1);
    #if 0
    _beta.load_from_pf(_env.datfname, _k);
    _beta.set_next_to_prior(); 
    _beta.compute_expectations(); 
    if(_env.normalized_representations)
      _beta.mean_curr().normalize1();
    #endif

  } else {
    for(uint32_t t=0; t<_thetas.size(); ++t) {
      // initialization is the same across timesteps
      gsl_rng_set(_r, _env.seed);
      _thetas[t]->initialize();

      if(_env.normalized_representations)
        _thetas[t]->mean_curr().normalize1();
      _thetas[t]->compute_expectations();

      // initialization is the same across timesteps (and different than for items)
      gsl_rng_set(_r, _env.seed+1);
      _betas[t]->initialize();

      if(_env.normalized_representations)
        _betas[t]->mean_curr().normalize1();
      _betas[t]->compute_expectations();

    }

    _theta.initialize();
    _theta.compute_expectations();

    _beta.initialize();
    _beta.compute_expectations();

    //if(_env.normalized_representations)
    //  _beta.mean_curr().normalize1();
    //_beta.compute_expectations();
  }

  if (_env.pf_init_static)
    load_correction_factors();

  printf("+ variables initialized\n"); 
}

void 
DynNormPRec::infer_dui_correction()
{
  lerr("running duv inference with correction");
  initialize();

  uint32_t x;
  x = _k;

  Array *p = new Array(x);
  Array *q = new Array(x);

  double * prior_array_thetas = new double[_k];
  double * prior_array_theta = new double[_k];
  double * prior_array_betas = new double[_k];
  double * prior_array_beta = new double[_k];
  for(uint32_t k=0; k<_k; ++k)  {
    prior_array_thetas[k] = _thetas[0]->mprior();
    prior_array_betas[k] = _betas[0]->mprior();
    prior_array_theta[k] = _theta.mprior();
    prior_array_beta[k] = _beta.mprior();
  }
  bool begin, end;

  Array ones(_k);
  for (uint32_t k=0; k<_k; ++k)
    ones[k] = 1.;

  compute_precision(false);
  //elbo();
  printf("\n");

  Matrix other_theta(_n, _k, true);
  double **otd = other_theta.data();
  Matrix other_beta(_m, _k, true);
  double **obd = other_beta.data();

  const double ** bd = _beta.expected_expv().const_data();
  const double ** td = _theta.expected_expv().const_data();

  bool optimize_correction = true, optimize_global = true;

  //explain_opt(optimize_correction, optimize_global); 

  while (1) {

    if (optimize_global) {
      Matrix phi_m(_m,x), phi_n(_n,x);

      phi_n.zero(); other_theta.zero();
      for (uint32_t t = 0; t <= _env.max_train_time_period; ++t) {
        begin=false; end=false;
        if (t == 0)
          begin=true;
        else if (t==_env.max_train_time_period)
          end=true;

#pragma omp parallel for num_threads(_env.num_threads)
        for (uint32_t n = 0; n < _n; ++n) { // for every user
          Array phi(x);

          const vector<uint32_t> *movies = _ratings.get_movies(n,t);
          for (uint32_t j = 0; j < movies->size(); ++j) {
            uint32_t m = (*movies)[j];
            yval_t y = _ratings.r(n,m,t);

            get_phi(*_thetas[t], n, *_betas[t], m, _theta, _beta, phi);
            //get_phi(_theta, n, _beta, m, phi);

            if (y > 1)
              phi.scale(y);

#pragma omp critical
            {
              phi_n.add_slice(n,phi);
            }

          } // items
        } // users

        const double ** btd = _betas[t]->expected_expv().const_data();
        Array tmp(_k); tmp.zero();
        double *tmpd = tmp.data();
        for(uint32_t m=0; m < _m; ++m)  {
          for(uint32_t k=0; k < _k; ++k) {
            tmpd[k] += bd[m][k] * btd[m][k];
          }
        }

        const double ** ttd = _thetas[t]->expected_expv().const_data();
        for(uint32_t n=0; n < _n; ++n)  {
          for(uint32_t k=0; k < _k; ++k) {
            otd[n][k] += ttd[n][k] * tmpd[k];
          }
        }

      } // time

#pragma omp parallel for num_threads(_env.num_threads)
      for (uint32_t n=0; n < _n; ++n) {
        Array p(x), ot(x);
        phi_n.slice(0, n, p);
        other_theta.slice(0, n, ot);
        _theta.update_mean_next(n, p, ot.const_data(), prior_array_theta, NULL, true, true, 1);
        _theta.update_var_next(n, ot.const_data());
      }
      _theta.swap();
      //_theta.swap_var();
      //_theta.swap_mean();
      _theta.compute_expectations();

      phi_m.zero(); other_beta.zero();
      for (uint32_t t = 0; t <= _env.max_train_time_period; ++t) {
        begin=false; end=false;
        if (t == 0)
          begin=true;
        else if (t==_env.max_train_time_period)
          end=true;

#pragma omp parallel for num_threads(_env.num_threads)
        for (uint32_t n = 0; n < _n; ++n) { // for every user
          Array phi(x);

          const vector<uint32_t> *movies = _ratings.get_movies(n,t);
          for (uint32_t j = 0; j < movies->size(); ++j) {
            uint32_t m = (*movies)[j];
            yval_t y = _ratings.r(n,m,t);

            get_phi(*_thetas[t], n, *_betas[t], m, _theta, _beta, phi);
            //get_phi(_theta, n, _beta, m, phi);

            if (y > 1)
              phi.scale(y);

#pragma omp critical
            {
              phi_m.add_slice(m, phi);
            }

          } // items
        } // users

        const double ** ttd = _thetas[t]->expected_expv().const_data();
        Array tmp(_k); tmp.zero();
        double *tmpd = tmp.data();
        for(uint32_t n=0; n < _n; ++n)  {
          for(uint32_t k=0; k < _k; ++k) {
            tmpd[k] += td[n][k] * ttd[n][k];
          }
        }

        const double ** btd = _betas[t]->expected_expv().const_data();
        for(uint32_t m=0; m < _m; ++m)  {
          for(uint32_t k=0; k < _k; ++k) {
            obd[m][k] += btd[m][k] * tmpd[k];
          }
        }
      } // time

#pragma omp parallel for num_threads(_env.num_threads)
      for (uint32_t m=0; m < _m; ++m) {
        Array p(x), ob(x);
        phi_m.slice(0, m, p);
        other_beta.slice(0, m, ob);

        _beta.update_mean_next(m, p, ob.const_data(), prior_array_beta, NULL, true, true, 1);
        _beta.update_var_next(m, ob.const_data());
      }
      _beta.swap();
      //_beta.swap_var();
      //_beta.swap_mean();
      _beta.compute_expectations();
    }

  if (optimize_correction) {
    // a loop over time periods (this code doesn't have to know
    // what a time period means, i.e., we assume discrete time for now)
    for (uint32_t t = 0; t <= _env.max_train_time_period; ++t) {
      begin=false; end=false;
      if (t == 0)
        begin=true;
      else if (t==_env.max_train_time_period)
        end=true;

      other_theta.zero();
      other_beta.zero();

#pragma omp parallel for num_threads(_env.num_threads)
      for (uint32_t n = 0; n < _n; ++n) { // for every user
        //if (t > _ratings.last_user_time_period(n))
        //  continue;
        Array phi_n(x), phi(x);
        phi_n.zero();
        const vector<uint32_t> *movies = _ratings.get_movies(n,t);
        for (uint32_t j = 0; j < movies->size(); ++j) {
          uint32_t m = (*movies)[j];
          yval_t y = _ratings.r(n,m,t);

          get_phi(*_thetas[t], n, *_betas[t], m, _theta, _beta, phi);

          if (y > 1)
            phi.scale(y);

          phi_n.add_to(phi);

        } // items

        const double ** btd = _betas[t]->expected_expv().const_data();

        Array tmp(_k); tmp.zero();
        double *tmpd = tmp.data();
        for (uint32_t m=0; m < _m; ++m)
          for (uint32_t k=0; k < _k; ++k)
            tmpd[k] += bd[m][k] * btd[m][k];

        // add current theta
        const double **td = _theta.expected_expv().const_data();
        for (uint32_t k=0; k<_k; ++k)
          otd[n][k] = tmpd[k] * td[n][k];

        if(!_env.normalized_representations) {
          if(begin) {
            _thetas[t]->update_mean_next(n, phi_n, otd[n], prior_array_thetas, _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          } else if(end) {
            _thetas[t]->update_mean_next(n, phi_n, otd[n], _thetas[t-1]->mean_curr().const_data()[n], NULL, begin, end, 1);
          } else {
            _thetas[t]->update_mean_next(n, phi_n, otd[n], _thetas[t-1]->mean_curr().const_data()[n], _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          }
        } else { // normalized_representations
          if(begin) {
            _thetas[t]->update_norm_mean_next(n, phi_n, otd[n], prior_array_thetas, _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          } else if(end) {
            _thetas[t]->update_norm_mean_next(n, phi_n, otd[n], _thetas[t-1]->mean_curr().const_data()[n], NULL, begin, end, 1);
          } else {
            _thetas[t]->update_norm_mean_next(n, phi_n, otd[n], _thetas[t-1]->mean_curr().const_data()[n], _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          }
        } // normalized_representations
        _thetas[t]->update_var_next(n, otd[n], t == _env.max_train_time_period ? true: false, 1);

      } // users

      _thetas[t]->swap();
      _thetas[t]->compute_expectations();

    } // time


    for (uint32_t t = 0; t <= _env.max_train_time_period; ++t) {
      begin=false; end=false;
      if (t == 0)
        begin=true;
      else if (t==_env.max_train_time_period)
        end=true;

#pragma omp parallel for num_threads(_env.num_threads)
      for (uint32_t m = 0; m < _m; ++m) { // for every item
        Array phi_m(x), phi(x);
        phi_m.zero();

        const vector<uint32_t> *users = _ratings.get_users(m,t);
        for (uint32_t j = 0; j < users->size(); ++j) {
          uint32_t n = (*users)[j];
          yval_t y = _ratings.r(n,m,t);

          get_phi(*_thetas[t], n, *_betas[t], m, _theta, _beta, phi);

          if (y > 1)
            phi.scale(y);
          phi_m.add_to(phi);

        } // users

        const double ** ttd = _thetas[t]->expected_expv().const_data();

        Array tmp(_k); tmp.zero();
        double *tmpd = tmp.data();
        for (uint32_t n=0; n < _n; ++n)
          for (uint32_t k=0; k < _k; ++k)
            tmpd[k] += td[n][k] * ttd[n][k];

        // add current theta
        const double **bd = _beta.expected_expv().const_data();
        for (uint32_t k=0; k<_k; ++k)
          obd[m][k] = tmpd[k] * bd[m][k];

        if(!_env.normalized_representations) {
          if(begin) {
            _betas[t]->update_mean_next(m, phi_m, obd[m], prior_array_betas, _betas[t+1]->mean_curr().const_data()[m], begin, end, 1);
          } else if(end) {
            _betas[t]->update_mean_next(m, phi_m, obd[m], _betas[t-1]->mean_curr().const_data()[m], NULL, begin, end, 1);
          } else {
            _betas[t]->update_mean_next(m, phi_m, obd[m], _betas[t-1]->mean_curr().const_data()[m], _betas[t+1]->mean_curr().const_data()[m], begin, end, 1);
          }
        } else { // normalized_representations
          if(begin) {
            _betas[t]->update_norm_mean_next(m, phi_m, obd[m], prior_array_betas, _betas[t+1]->mean_curr().const_data()[m], begin, end, 1);
          } else if(end) {
            _betas[t]->update_norm_mean_next(m, phi_m, obd[m], _betas[t-1]->mean_curr().const_data()[m], NULL, begin, end, 1);
          } else {
            _betas[t]->update_norm_mean_next(m, phi_m, obd[m], _betas[t-1]->mean_curr().const_data()[m], _betas[t+1]->mean_curr().const_data()[m], begin, end, 1);
          }
        } // normalized_representations
        _betas[t]->update_var_next(m, obd[m], t == _env.max_train_time_period ? true: false, 1);

      } // items

      _betas[t]->swap();
      _betas[t]->compute_expectations();

    } // time
  }

    printf("\r iteration %d (%d s)", _iter, duration());
    fflush(stdout);
    if (_iter % _env.reportfreq == 0) {
      compute_likelihood(true);
      compute_likelihood(false);
      compute_precision(false);
      save_model();
      //elbo();
      printf("\n");
    }

    if (_env.save_state_now) {
      lerr("Saving state at iteration %d duration %d secs", _iter, duration());
      do_on_stop();
    }

    if (_iter == 10) {
      //printf("now optimizing correction parameters only\n");
      optimize_global = true;
      optimize_correction = true;
      //explain_opt(optimize_correction, optimize_global); 
      //_env.reportfreq=1;
    }

    _iter++;
  }
  delete p;
  delete q;
  delete prior_array_thetas; delete prior_array_theta;
  delete prior_array_betas; delete prior_array_beta;
}

void
DynNormPRec::infer_dui()
{
  lerr("running duv inference()");
  initialize();

  uint32_t x;
  x = _k;

  //Array phi(x); 
  //Array phi_n(x), phi_m(x);
  Array *p = new Array(x);
  Array *q = new Array(x);

  double * prior_array_theta = new double[_k];
  double * prior_array_beta = new double[_k];
  for(uint32_t k=0; k<_k; ++k)  { 
    prior_array_theta[k] = _thetas[0]->mprior();
    prior_array_beta[k] = _betas[0]->mprior();
  }
  bool begin, end;

  //Array betaexpsum(_k);
  //Array thetaexpsum(_k);

  Matrix thetasummeanb(_n,_k); 
  Matrix thetasummeane(_n,_k); 

  Array ones(_k); 
  for (uint32_t k=0; k<_k; ++k)
    ones[k] = 1.; 

  compute_precision(false);
  //elbo(); 
  printf("\n"); 

  while (1) {

    // a loop over time periods (this code doesn't have to know
    // what a time period means, i.e., we assume discrete time for now) 
    for (uint32_t t = 0; t <= _env.max_train_time_period; ++t) {
      begin=false; end=false;
      if (t == 0)
        begin=true; 
      else if (t==_env.max_train_time_period)
        end=true; 

      #pragma omp parallel for num_threads(_env.num_threads)
      for (uint32_t n = 0; n < _n; ++n) { // for every user 
        //if (t > _ratings.last_user_time_period(n))
        //  continue;
        Array phi_n(x), phi(x);
        phi_n.zero(); 
        const vector<uint32_t> *movies = _ratings.get_movies(n,t);
        for (uint32_t j = 0; j < movies->size(); ++j) {
          uint32_t m = (*movies)[j];
          yval_t y = _ratings.r(n,m,t); 

          get_phi(*_thetas[t], n, *_betas[t], m, phi);

          if (y > 1)
            phi.scale(y);

          phi_n.add_to(phi); 

        } // items

        Array betaexpsum(_k);
        betaexpsum.zero();

        _betas[t]->sum_eexp_rows(betaexpsum);
        
        if(!_env.normalized_representations) {
          if(begin) {
            _thetas[t]->update_mean_next(n, phi_n, betaexpsum.const_data(), prior_array_theta, _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          } else if(end) {
            _thetas[t]->update_mean_next(n, phi_n, betaexpsum.const_data(), _thetas[t-1]->mean_curr().const_data()[n], NULL, begin, end, 1);
          } else {
            _thetas[t]->update_mean_next(n, phi_n, betaexpsum.const_data(), _thetas[t-1]->mean_curr().const_data()[n], _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          }
        } else { // normalized_representations
          if(begin) {
            _thetas[t]->update_norm_mean_next(n, phi_n, betaexpsum.const_data(), prior_array_theta, _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          } else if(end) {
            _thetas[t]->update_norm_mean_next(n, phi_n, betaexpsum.const_data(), _thetas[t-1]->mean_curr().const_data()[n], NULL, begin, end, 1);
          } else {
            _thetas[t]->update_norm_mean_next(n, phi_n, betaexpsum.const_data(), _thetas[t-1]->mean_curr().const_data()[n], _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          }
        } // normalized_representations

        _thetas[t]->update_var_next(n, betaexpsum.const_data(), t == _env.max_train_time_period ? true: false, 1); 

      } // users

      _thetas[t]->swap(); 
      _thetas[t]->compute_expectations(); 

    } // time

    if (!_env.fixed_item_param) {

      for (uint32_t t = 0; t <= _env.max_train_time_period; ++t) { 
        begin=false; end=false;
        if (t == 0)
          begin=true; 
        else if (t==_env.max_train_time_period)
          end=true; 

        #pragma omp parallel for num_threads(_env.num_threads)
        for (uint32_t m = 0; m < _m; ++m) { // for every item 
          Array phi_m(x), phi(x);
          phi_m.zero(); 

          const vector<uint32_t> *users = _ratings.get_users(m,t);
          for (uint32_t j = 0; j < users->size(); ++j) {
            uint32_t n = (*users)[j];
            yval_t y = _ratings.r(n,m,t); 

            get_phi(*_thetas[t], n, *_betas[t], m, phi);

            if (y > 1)
              phi.scale(y);
            phi_m.add_to(phi); 

          } // users

          Array thetaexpsum(_k); 
          thetaexpsum.zero();
          _thetas[t]->sum_eexp_rows(thetaexpsum);

          if(!_env.normalized_representations) {
            if(begin) {
              _betas[t]->update_mean_next(m, phi_m, thetaexpsum.const_data(), prior_array_beta, _betas[t+1]->mean_curr().const_data()[m], begin, end, 1);
            } else if(end) {
              _betas[t]->update_mean_next(m, phi_m, thetaexpsum.const_data(), _betas[t-1]->mean_curr().const_data()[m], NULL, begin, end, 1);
            } else {
              _betas[t]->update_mean_next(m, phi_m, thetaexpsum.const_data(), _betas[t-1]->mean_curr().const_data()[m], _betas[t+1]->mean_curr().const_data()[m], begin, end, 1);
            }
          } else { // normalized_representations
            if(begin) {
              _betas[t]->update_norm_mean_next(m, phi_m, thetaexpsum.const_data(), prior_array_beta, _betas[t+1]->mean_curr().const_data()[m], begin, end, 1);
            } else if(end) {
              _betas[t]->update_norm_mean_next(m, phi_m, thetaexpsum.const_data(), _betas[t-1]->mean_curr().const_data()[m], NULL, begin, end, 1);
            } else {
              _betas[t]->update_norm_mean_next(m, phi_m, thetaexpsum.const_data(), _betas[t-1]->mean_curr().const_data()[m], _betas[t+1]->mean_curr().const_data()[m], begin, end, 1);
            }
          } // normalized_representations

          _betas[t]->update_var_next(m, thetaexpsum.const_data(), t == _env.max_train_time_period ? true: false, 1); 

        } // items

        _betas[t]->swap(); 
        _betas[t]->compute_expectations(); 

      } // time 

    }

    printf("\r iteration %d (%d s)", _iter, duration());
    fflush(stdout);    
    if (_iter % _env.reportfreq == 0) {
      compute_likelihood(true);
      compute_likelihood(false);
      save_model();
      compute_precision(false);
      //elbo(); 
      printf("\n"); 
    }

    if (_env.save_state_now) {
      lerr("Saving state at iteration %d duration %d secs", _iter, duration());
      do_on_stop();
    }

    _iter++;
  }
  delete p; 
  delete q;
  delete prior_array_theta;
  delete prior_array_beta;
}

// perform inference
void
DynNormPRec::infer() 
{
  lerr("running inference()");
  initialize();

  uint32_t x;
  x = _k; 

  Matrix *phi_m = new Matrix(_m,x); 
  Array phi(x); 
  Array phi_n(x);
  Array *p = new Array(x);
  Array *q = new Array(x);

  double * prior_array = new double[_k];
  double * prior_array_beta = new double[_k];
  for(uint32_t k=0; k<_k; ++k)  { 
    prior_array[k] = _thetas[0]->mprior();
    prior_array_beta[k] = _beta.mprior();
  }
  bool begin, end; 

  //vector<unordered_set<uint32_t>> user_rated;
  Array betaexpsum(_k);
  Matrix thetasum_mat(_m, _k, true);
  Array thetaexpsum(_k);

  Matrix thetasummeanb(_n,_k); 
  Matrix thetasummeane(_n,_k); 

  Array ones(_k); 
  for (uint32_t k=0; k<_k; ++k)
    ones[k] = 1.; 

  compute_precision(false);
  //elbo(); 
  printf("\n"); 

  while (1) {

    thetaexpsum.zero();

    // a loop over time periods (this code doesn't have to know
    // what a time period means, i.e., we assume discrete time for now) 
    for (uint32_t t = 0; t <= _env.max_train_time_period; ++t) {
      begin=false; end=false;
      if (t == 0)
        begin=true; 
      else if (t==_env.max_train_time_period)
        end=true; 

      for (uint32_t n = 0; n < _n; ++n) { // for every user 

        //if(_env.positive_become_unobs) {
        //  rated.clear(); 
        //}

        phi_n.zero(); 
        const vector<uint32_t> *movies = _ratings.get_movies(n,t);
        //printf("movies size %d\n", movies->size()); 
        for (uint32_t j = 0; j < movies->size(); ++j) {
          uint32_t m = (*movies)[j];
          yval_t y = _ratings.r(n,m,t); 

          get_phi(*_thetas[t], n, _beta, m, phi);

          if (y > 1)
            phi.scale(y);

#if 0
          const double * ppd = phi.const_data(); 
          printf("\n in update (%d-%d)\n", n, m); 
          for(uint32_t kk=0;kk<_k;++kk)
            printf("%f ", ppd[kk]); 
          printf("\n"); 
#endif

          phi_n.add_to(phi); 

          //if(_env.positive_become_unobs) { 
          //// remove this item's contribution from betaexpsum 
          //
          // 

        }

        if(!_env.normalized_representations) {
          if(begin) {
            _thetas[t]->update_mean_next(n, phi_n, betaexpsum.const_data(), prior_array, _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          } else if(end) {
            _thetas[t]->update_mean_next(n, phi_n, betaexpsum.const_data(), _thetas[t-1]->mean_curr().const_data()[n], NULL, begin, end, 1);
          } else {
            _thetas[t]->update_mean_next(n, phi_n, betaexpsum.const_data(), _thetas[t-1]->mean_curr().const_data()[n], _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          }
        } else { // normalized_representations
          if(begin) {
            _thetas[t]->update_norm_mean_next(n, phi_n, betaexpsum.const_data(), prior_array, _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          } else if(end) {
            _thetas[t]->update_norm_mean_next(n, phi_n, betaexpsum.const_data(), _thetas[t-1]->mean_curr().const_data()[n], NULL, begin, end, 1);
          } else {
            _thetas[t]->update_norm_mean_next(n, phi_n, betaexpsum.const_data(), _thetas[t-1]->mean_curr().const_data()[n], _thetas[t+1]->mean_curr().const_data()[n], begin, end, 1);
          }
        } // normalized_representations

        _thetas[t]->update_var_next(n, betaexpsum.const_data(), t == _env.max_train_time_period ? true: false, 1); 

      } // users

      if (t == 0)
        _thetas[t]->swap(); 
      else
        _thetas[t]->swap(); 
        //_thetas[t]->swap(_thetas[t-1]->mean_curr()); 
      _thetas[t]->compute_expectations(); 

    } // time

    if (!_env.fixed_item_param) {

      phi_m->zero(); 
      thetaexpsum.zero();
      for (uint32_t t = 0; t <= _env.max_train_time_period; ++t) { 

        _thetas[t]->sum_eexp_rows(thetaexpsum);

        for (uint32_t n = 0; n < _n; ++n) { // for every user 

          const vector<uint32_t> *movies = _ratings.get_movies(n,t);
          for (uint32_t j = 0; j < movies->size(); ++j) {
            uint32_t m = (*movies)[j];
            yval_t y = _ratings.r(n,m,t); 

            get_phi(*_thetas[t], n, _beta, m, phi);

            if (y > 1)
              phi.scale(y);
            phi_m->add_slice(m,phi);
          }
        } // users
      } // time 

      const double * thetasumd = thetaexpsum.const_data();

      for (uint32_t m = 0; m < _m; ++m) { // for every item
        phi_m->slice(0, m, *p);
        _beta.update_mean_next(m, *p, thetasumd, prior_array_beta, NULL, true, true, 1);
        _beta.update_var_next(m, thetasumd); 
      }
      _beta.swap(); 
      _beta.compute_expectations(); 
    }

    printf("\r iteration %d", _iter);
    fflush(stdout);    
    if (_iter % _env.reportfreq == 0) {
      compute_likelihood(true);
      compute_likelihood(false);
      save_model();
      compute_precision(false);
      //elbo(); 
      printf("\n"); 
    }

    if (_env.save_state_now) {
      lerr("Saving state at iteration %d duration %d secs", _iter, duration());
      do_on_stop();
    }

    _iter++;
  }
  delete p; 
  delete q;
  delete phi_m;
  delete prior_array;
  delete prior_array_beta;
}



void
DynNormPRec::save_model()
{
  if (_env.dynamic_user_and_item_representations) {
    #pragma omp parallel for num_threads(_env.num_threads)
    for (uint32_t t = 0; t <= _env.max_train_time_period; ++t)
     _betas[t]->save_state(_ratings.seq2movie());
  }
  _beta.save_state(_ratings.seq2movie());

  #pragma omp parallel for num_threads(_env.num_threads)
  for (uint32_t t = 0; t <= _env.max_train_time_period; ++t)
    _thetas[t]->save_state(_ratings.seq2user());
  _theta.save_state(_ratings.seq2user());

}

void
DynNormPRec::compute_precision(bool save_ranking_file)
{
    double mhits10 = 0, mhits100 = 0;
    double cumndcg10 = 0, cumndcg100 = 0;
    uint32_t total_users = 0;
    FILE *f = 0;
    if (save_ranking_file)
        f = fopen(Env::file_str("/ranking.tsv").c_str(), "w");

    if (!save_ranking_file) {
        _sampled_users.clear();
        do {
            uint32_t n = gsl_rng_uniform_int(_r, _n);
            _sampled_users[n] = true;
        } while (_sampled_users.size() < 1000 && _sampled_users.size() < _n / 2);
    } 

    KVArray mlist(_m);
    KVIArray ndcglist(_m);

    uint32_t cc = 0; 
    for (UserMap::const_iterator itr = _sampled_users.begin();
            itr != _sampled_users.end(); ++itr, ++cc) {

        if(save_ranking_file)
          printf("\r%d/%lu", cc, _sampled_users.size()); 
        uint32_t n = itr->first;

        // for each user we use the latest "trained" time period as the one to predict with 
        uint32_t t = _ratings.last_user_time_period(n);
        lerr("User %d establishing ranking at time %d\n", n, t); 

        for (uint32_t m = 0; m < _m; ++m) {
            //Rating r(n,m,t);
            if (_ratings.allr(n,m) > 0 || is_validation(n,m,t)) { // skip training and validation
                mlist[m].first = m;
                mlist[m].second = .0;
                ndcglist[m].first = m;
                ndcglist[m].second = 0; 
                continue;
            }
            double u = .0;
            u = prediction_score(n, m, t);
            if (std::isnan(u))
              printf("u is nan (%d,%d,%d)\n", n, m, t); 
            mlist[m].first = m;
            mlist[m].second = u;
            ndcglist[m].first = m;
            CountMap::const_iterator itr;
            for(uint32_t it=0; it<_env.time_periods; ++it) {
                itr = _test_map.find(Rating(n, m, it));
                if(itr != _test_map.end())
                    break;
            }

            if (itr != _test_map.end()) {
                ndcglist[m].second = itr->second;
            } else { 
                ndcglist[m].second = 0;
            }      
        }
        uint32_t hits10 = 0, hits100 = 0;
        double   dcg10 = .0, dcg100 = .0; 
        mlist.sort_by_value();

        //uint32_t testhits = 0; 
        for (uint32_t j = 0; j < mlist.size() && j < _topN_by_user; ++j) {
            KV &kv = mlist[j];
            uint32_t m = kv.first;
            double pred = kv.second;
            Rating r(n, m, t);

            uint32_t m2 = 0, n2 = 0;
            if (save_ranking_file) {
                IDMap::const_iterator it = _ratings.seq2user().find(n);
                assert (it != _ratings.seq2user().end());

                IDMap::const_iterator mt = _ratings.seq2movie().find(m);
                if (mt == _ratings.seq2movie().end())
                    continue;

                m2 = mt->second;
                n2 = it->second;
            }

            // finds if user has this rating anywhere in the future
            // TODO: really what we should do is to come up with a ranking
            // for any time period the user has heldout data for.
            // Then average the precision given that ranking across all
            // time periods. (anything after the last train example should
            // be lumped into a single time period)
            CountMap::const_iterator itr;
            for(uint32_t it=t; it<_env.time_periods; ++it) {
                itr = _test_map.find(Rating(n, m, it));
                if(itr != _test_map.end())
                    break;
            }

            if (itr != _test_map.end()) {
                int v_ = itr->second;
                int v;
                if (_ratings.test_hit(v_))
                    { v = 1;}//testhits++;
                else
                    v = 0;

                if (j < 10) {
                    if (v > 0) { //hit
                        hits10++;
                        hits100++;
                    }
                    if (v_ > 0) { //has non-zero relevance
                        dcg10 += (pow(2.,v_) - 1)/log(j+2);
                        dcg100 += (pow(2.,v_) - 1)/log(j+2);
                    }
                } else if (j < 100) {
                    if (v > 0)
                        hits100++;
                    if (v_ > 0)
                        dcg100 += (pow(2.,v_) - 1)/log(j+2);
                }

                if (save_ranking_file) {
                    if (_ratings.allr(n, m) == .0)  {
                        //double hol = rating_likelihood(n,m,t,v);
                        //fprintf(f, "%d\t%d\t%.5f\t%d\t%.5f\n", n2, m2, pred, v,
                        //(pow(2.,v_) - 1)/log(j+2));
                        fprintf(f, "%d\t%d\t%.5f\t%d\n", n2, m2, pred, v);
                    }
                }
            } else {
                if (save_ranking_file) {
                    if (_ratings.allr(n, m) == .0) {
                        //double hol = rating_likelihood(n,m,t,0);
                        //fprintf(f, "%d\t%d\t%.5f\t%d\t%.5f\n", n2, m2, pred, 0, .0);
                        fprintf(f, "%d\t%d\t%.5f\t%d\n", n2, m2, pred, 0);
                    }
                }
            }
        }
        uint32_t idm;
        IDMap::const_iterator idt = _ratings.seq2user().find(n);
        if (idt != _ratings.seq2user().end()) 
          idm = idt->second;
        else
          idm = 0;
        //printf("test hits %d (user %d), time period (%d)\n", testhits, idm, t); 
        mhits10 += (double)hits10 / 10;
        if (std::isnan(mhits10))
          printf("mhits10 is nan\n"); 

        mhits100 += (double)hits100 / 100;
        total_users++;
        // DCG normalizer
        double dcg10_gt = 0, dcg100_gt = 0;
        bool user_has_test_ratings = true; 
        ndcglist.sort_by_value();
        for (uint32_t j = 0; j < ndcglist.size() && j < _topN_by_user; ++j) {
            int v = ndcglist[j].second; 
            if(v==0) { //all subsequent docs are irrelevant
                if(j==0)
                    user_has_test_ratings = false; 
                break;
            }

            if (j < 10) { 
                dcg10_gt += (pow(2.,v) - 1)/log(j+2);
                dcg100_gt += (pow(2.,v) - 1)/log(j+2);
            } else if (j < 100) {
                dcg100_gt += (pow(2.,v) - 1)/log(j+2);
            }
        }
        if(user_has_test_ratings) { 
            cumndcg10 += dcg10/dcg10_gt;
            cumndcg100 += dcg100/dcg100_gt;
        } 
    }
    if (save_ranking_file)
        fclose(f);
    fprintf(_pf, "%d\t%.5f\t%.5f\n", 
            total_users,
            (double)mhits10 / total_users, 
            (double)mhits100 / total_users);
    fflush(_pf);
    fprintf(_df, "%.5f\t%.5f\n", 
            cumndcg10 / total_users, 
            cumndcg100 / total_users);
    fflush(_df);

    printf("\n\t\t\tprec@10: %f, prec@%d: %f", (double)mhits10 / total_users, 100, (double)mhits100 / total_users); 

}

void
DynNormPRec::get_phi(NormBase<Matrix> &a, uint32_t ai, 
               NormBase<Matrix> &b, uint32_t bi, 
               Array &phi)
{

  assert (phi.size() == a.k() &&
	  phi.size() == b.k());
  assert (ai < a.n() && bi < b.n());
  const double  **ea = a.expected_v().const_data();
  const double  **eb = b.expected_v().const_data();
  for (uint32_t k = 0; k < _k; ++k)
    phi[k] = ::exp(ea[ai][k] + eb[bi][k] -1);
  phi.normalize();

}


void
DynNormPRec::get_phi(NormBase<Matrix> &a, uint32_t ai, 
		 NormBase<Matrix> &b, uint32_t bi, 
     NormBase<Matrix> &c, NormBase<Matrix> &d,
		 Array &phi)
{
  assert (phi.size() == a.k() &&
	  phi.size() == b.k());
  assert (ai < a.n() && bi < b.n());
  const double  **ea = a.expected_v().const_data();
  const double  **eb = b.expected_v().const_data();
  const double  **ec = c.expected_v().const_data();
  const double  **ed = d.expected_v().const_data();
  for (uint32_t k = 0; k < _k; ++k)
    phi[k] = ::exp(ea[ai][k] + ec[ai][k] + eb[bi][k] + ed[bi][k] -1);
  phi.normalize();
}

void
DynNormPRec::get_phi(NormBase<Matrix> &a, uint32_t ai, 
		 NormBase<Matrix> &b, uint32_t bi, 
		 double biasa, double biasb,
		 Array &phi)
{
  assert (phi.size() == a.k() + 2 &&
	  phi.size() == b.k() + 2);
  assert (ai < a.n() && bi < b.n());
  const double  **ea = a.expected_v().const_data();
  const double  **eb = b.expected_v().const_data();
  //phi.zero();
  for (uint32_t k = 0; k < _k; ++k)
    phi[k] = ::exp(ea[ai][k] + eb[bi][k] -1);
  phi[_k] = biasa;
  phi[_k+1] = biasb;
  phi.normalize();
}


void
DynNormPRec::compute_likelihood(bool validation)
{
  uint32_t k = 0; 
  double s = .0;
  
  CountMap *mp = NULL;
  FILE *ff = NULL;
  if (validation) {
    mp = &_validation_map;
    ff = _vf;
  } else {
    mp = &_test_map;
    ff = _tf;
  }

  for (CountMap::const_iterator i = mp->begin();
       i != mp->end(); ++i) {
    const Rating &e = i->first;
    uint32_t n = std::get<0>(e);
    uint32_t m = std::get<1>(e);
    uint8_t  t = std::get<2>(e); 

    yval_t r = i->second;
    double u = rating_likelihood(n,m,t,r);
    lerr("validation time-step %d\n", t); 

    s += u;
    k += 1;

    // add a fake 0 
    //for(uint32_t ii=0; ii<5; ++ii) {
      //s += rating_likelihood(n, gsl_rng_uniform_int(_r, _m), t, 0); 
      //k += 1; 
    //}
  }

  double a = .0;
  a = s / k;  
  info("s = %.5f\n", s);
  fprintf(ff, "%d\t%d\t%.9f\t%d\n", _iter, duration(), a, k);
  fflush(ff);

  if (!validation)
    return;

  printf("\n\t\t\tvalidation pred ll: %f", a);
  
  bool stop = false;
  int why = -1;
  if (_iter > 30) {
    if (a > _prev_h && _prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.000001) {
      stop = true;
      why = 0;
    } else if (a < _prev_h)
      _nh++;
    else if (a > _prev_h)
      _nh = 0;

    if (_nh > 2) { // be robust to small fluctuations in predictive likelihood
      why = 1;
      stop = true;
    }
  }
  _prev_h = a;
  FILE *f = fopen(Env::file_str("/max.txt").c_str(), "w");
  fprintf(f, "%d\t%d\t%.5f\t%d\n", 
	  _iter, duration(), a, why);
  fclose(f);
  if (stop) {
    do_on_stop();
    exit(0);
  }
}

double
DynNormPRec::score(uint32_t p, uint32_t q, uint32_t t) const
{
  if(t == (_env.time_periods-1)) { // !! hack for arXiv data (most validation/test occurs one step after last train)
    t -= 1; 
    lerr("predicting (%d,%d) with time %d\n", p, q, t); 
  }
  const double **eexpthetas = _thetas[t]->expected_expv().const_data();
  const double **eexptheta = _theta.expected_expv().const_data();
  const double **eexpbeta = NULL, **eexpbetas = NULL;

  if (_env.dynamic_user_and_item_representations) {
    eexpbetas = _betas[t]->expected_expv().const_data();
    eexpbeta = _beta.expected_expv().const_data();
  } else
    eexpbetas = _beta.expected_expv().const_data();
  
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k) 
    //s += (eexptheta[p][k] * eexpbeta[q][k]);
    s += (eexptheta[p][k] * eexpthetas[p][k] * eexpbeta[q][k] * eexpbetas[q][k]);

 
  if (s < 1e-15)
    s = 1e-15;

  return s; 

}


double
DynNormPRec::rating_likelihood(uint32_t p, uint32_t q, uint32_t t, yval_t y) const
{

  // predict with the latest time step for which there's train data
  // wrongly assumes that each user has been trained up to that timestep
  uint32_t lastt = _ratings.last_user_time_period(p);
  if(t > lastt)
    t = lastt;

  double s = score(p,q,t);

  if(std::isinf(log(1 - exp(-s))) )
    printf("expression is inf %f\n", s); 
  if (_env.binary_data)
    return y == 0 ? -s : log(1 - exp(-s));    
  return y * log(s) - s - log_factorial(y);
}

void
DynNormPRec::do_on_stop()
{
  save_model();
  gen_ranking_for_users(false);
}

void
DynNormPRec::load_correction_factors()
{
  // loads thetas,beta (typically from previous run of the same model)
  string nn = ""; 
  _theta.load(nn); // sets curr mean/var to pf fit
  //_env.datfname
  _theta.set_next_to_prior(); // sets next mean/var to corresponding priors
  _theta.compute_expectations();

  _beta.load(nn); // sets curr mean/var to pf fit
  _beta.set_next_to_prior();
  _beta.compute_expectations();
}

void
DynNormPRec::load_factors()
{
  for(uint32_t t=0; t<_thetas.size(); ++t) { 
    // loads thetas,beta (typically from previous run of the same model)
    _thetas[t]->load(); // sets curr mean/var to pf fit
    // sets next mean/var to corresponding priors
    _thetas[t]->set_next_to_prior(); 
    //if(t==0)
    //  _thetas[t]->set_next_to_prior(); 
    //else {
    //  _thetas[t]->set_mean_next(_thetas[t-1]->mean_curr());
    //  _thetas[t]->set_var_next(); 
    //}
  }

  for(uint32_t t=0; t<_betas.size(); ++t) { 
    _betas[t]->load(); // sets curr mean/var to pf fit
    _betas[t]->set_next_to_prior(); 
  }
}

void
DynNormPRec::gen_ranking_for_users(bool load)
{
  if (load) { 
    load_factors(); 
  } 

  char buf[4096];
  sprintf(buf, "%s/test_users.tsv", _env.datfname.c_str());
  FILE *f = fopen(buf, "r");
  if (!f) {
    lerr("cannot open %s", buf);
    printf("cannot open %s", buf); 
    return;
  }
  //assert(f);
  _sampled_users.clear();
  _ratings.read_test_users(f, &_sampled_users);
  fclose(f);
  compute_precision(true);
  printf("\nDONE writing ranking.tsv in output directory\n");
  fflush(stdout);
}

double
DynNormPRec::prediction_score(uint32_t p, uint32_t q, uint32_t t) const
{
  double s = score(p, q, t);
  assert(s>0);

  if (_use_rate_as_score)
    return s;
  
  if (s < 1e-30)
    s = 1e-30;
  double prob_zero = exp(-s);
  return 1 - prob_zero;
}
