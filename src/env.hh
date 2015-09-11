#ifndef ENV_HH
#define ENV_HH

#define __STDC_FORMAT_MACROS
#define __STDC_LIMIT_MACROS

#include <inttypes.h>

#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <map>
#include <list>
#include <vector>
#include <unordered_map>
#include "matrix.hh"
#include "log.hh"

typedef uint8_t yval_t;

typedef D2Array<yval_t> AdjMatrix;
typedef D2Array<double> Matrix;
typedef D3Array<double> D3;
typedef D2Array<KV> MatrixKV;
typedef D1Array<KV> KVArray;
typedef D1Array<KVI> KVIArray;

typedef std::map<uint32_t, yval_t> RatingMap;
typedef std::map<uint32_t, uint32_t> IDMap;
typedef std::map<uint32_t, uint32_t> FreqMap;
typedef std::map<string, uint32_t> FreqStrMap;
typedef std::map<string, uint32_t> StrMap;
typedef std::map<uint32_t, string> StrMapInv;

typedef D1Array<std::vector<uint32_t> *> SparseMatrix;
typedef D1Array<std::unordered_map<uint32_t, std::vector<uint32_t> *> *> SparseMatrixT; 
typedef D1Array<RatingMap *> SparseMatrixR;
typedef D1Array<std::map<uint32_t, RatingMap *> *> SparseMatrixTR; 
typedef std::vector<Rating> RatingList;
typedef std::map<uint32_t, bool> UserMap;
typedef std::map<uint32_t, bool> MovieMap;
typedef std::map<uint32_t, bool> BoolMap;
typedef std::map<uint32_t, double> DoubleMap;
typedef std::map<uint32_t, Array *> ArrayMap;
typedef std::map<uint32_t, uint32_t> ValMap;
typedef std::map<uint32_t, vector<uint32_t> > MapVec;
typedef MapVec SparseMatrix2;
typedef std::map<Rating, bool> SampleMap;
typedef std::map<Rating, int> CountMap;
typedef std::map<Rating, double> ValueMap;
typedef std::map<uint32_t, string> StrMapInv;

class Env {
public:
  typedef enum { CREATE_TRAIN_TEST_SETS, TRAINING } Mode;
  Env(uint32_t N, uint32_t M, uint32_t K, string fname, 
      uint32_t rfreq, double rseed,
      uint32_t max_iterations, bool load, string loc, 
      bool batch, bool binary_data, 
      uint32_t rating_threshold, bool normal_priors,
      bool fixed_item_param, bool pf_init, bool pf_init_static,
      double vprior, bool dynamic_item_representations,
      bool dynamic_user_and_item_representations, uint32_t num_threads, 
      uint32_t time_period_length);

  ~Env() { fclose(_plogf); }

  static string prefix;
  static Logger::Level level;

  uint32_t n;  // users
  uint32_t m;  // movies
  uint32_t k;
  uint32_t t;
  uint32_t mini_batch_size;

  double a;
  double b;
  double c;
  double d;

  double alpha;
  double tau0;
  double tau1;
  //double heldout_ratio;
  //double validation_ratio;
  int reportfreq;
  double epsilon;
  double logepsilon;
  bool strid;
  uint32_t max_iterations;
  double seed;
  bool save_state_now;
  string datfname;
  string label;
  bool nmi;
  string ground_truth_fname;
  bool model_load;
  string model_location;
  bool batch;
  Mode mode;
  bool binary_data;
  bool explore;
  uint32_t rating_threshold;
  bool normal_priors; 
  bool dynamic_item_representations;
  bool dynamic_user_and_item_representations;
  bool fixed_item_param;
  bool pf_init;
  bool pf_init_static;
  double vprior; 
  uint32_t num_threads;
  bool normalized_representations; 

  uint32_t time_periods;
  uint32_t time_period_length;
  uint32_t time_my_epoch;
  uint32_t max_train_time_period;

  template<class T> static void plog(string s, const T &v);
  static string file_str(string fname);
  // helper to set timestep information
  void read_for_stats(char *buf, uint32_t *min_rating_time, uint32_t *max_rating_time);

private:
  static FILE *_plogf;
};


template<class T> inline void
Env::plog(string s, const T &v)
{
  fprintf(_plogf, "%s: %s\n", s.c_str(), v.s().c_str());
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const double &v)
{
  fprintf(_plogf, "%s: %.9f\n", s.c_str(), v);
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const string &v)
{
  fprintf(_plogf, "%s: %s\n", s.c_str(), v.c_str());
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const bool &v)
{
  fprintf(_plogf, "%s: %s\n", s.c_str(), v ? "True": "False");
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const int &v)
{
  fprintf(_plogf, "%s: %d\n", s.c_str(), v);
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const unsigned &v)
{
  fprintf(_plogf, "%s: %d\n", s.c_str(), v);
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const short unsigned int &v)
{
  fprintf(_plogf, "%s: %d\n", s.c_str(), v);
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const uint64_t &v)
{
  fprintf(_plogf, "%s: %" PRIu64 "\n", s.c_str(), v);
  fflush(_plogf);
}

#ifdef __APPLE__
template<> inline void
Env::plog(string s, const long unsigned int &v)
{
  fprintf(_plogf, "%s: %lu\n", s.c_str(), v);
  fflush(_plogf);
}
#endif

inline string
Env::file_str(string fname)
{
  string s = prefix + fname;
  return s;
}

inline void 
Env::read_for_stats(char *buf, uint32_t *min_rating_time, uint32_t *max_rating_time)
{
  uint32_t mid = 0, uid = 0, rating = 0, rating_time = 0;
  *min_rating_time = UINT32_MAX;

  FILE *f = fopen(buf, "r"); 

  while (!feof(f)) {
    if (fscanf(f, "%u\t%u\t%u\t%u\n", &uid, &mid, &rating, &rating_time) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }
    if (rating_time < *min_rating_time)
      *min_rating_time = rating_time;
    if (rating_time > *max_rating_time)
      *max_rating_time = rating_time;
  }
  fclose(f); 
}


inline
Env::Env(uint32_t N, uint32_t M, uint32_t K, string fname, 
	 uint32_t rfreq, double rseed,
	 uint32_t maxitr, bool load, string loc,
	 bool batchv, bool binary_datav,
	 uint32_t rating_thresholdv, bool normal_priorsv,
     bool fixed_item_paramv, bool pf_initv, bool pf_init_staticv,
     double vpriorv, bool dynamic_item_representationsv,
     bool dynamic_user_and_item_representationsv, uint32_t num_threadsv,
     uint32_t time_period_lengthv)
  : n(N),
    m(M),
    k(K),
    t(2),
    mini_batch_size(1000),
    tau0(0),
    tau1(0),
    reportfreq(rfreq),
    epsilon(0.001),
    logepsilon(log(epsilon)),
    max_iterations(maxitr),
    seed(rseed),
    save_state_now(false),
    datfname(fname),
    model_load(load),
    model_location(loc),
    batch(batchv),
    mode(TRAINING),
    binary_data(binary_datav),
    rating_threshold(rating_thresholdv),
    normal_priors(normal_priorsv),
    fixed_item_param(fixed_item_paramv),
    pf_init(pf_initv),
    pf_init_static(pf_init_staticv),
    normalized_representations(false),
    vprior(vpriorv),
    dynamic_item_representations(dynamic_item_representationsv),
    dynamic_user_and_item_representations(dynamic_user_and_item_representationsv),
    num_threads(num_threadsv),
    time_period_length(time_period_lengthv)
{
  uint32_t max_rating_time=0; 
  char buf[4096];

  uint32_t valtest_epoch; 

  sprintf(buf, "%s/train.tsv", fname.c_str());
  read_for_stats(buf, &time_my_epoch, &max_rating_time); 

  sprintf(buf, "%s/validation.tsv", fname.c_str());
  read_for_stats(buf, &valtest_epoch, &max_rating_time); 
  if (valtest_epoch < time_my_epoch)
    time_my_epoch = valtest_epoch;

  sprintf(buf, "%s/test.tsv", fname.c_str());
  read_for_stats(buf, &valtest_epoch, &max_rating_time); 
  if (valtest_epoch < time_my_epoch)
    time_my_epoch = valtest_epoch;

  printf("+ max_rating_time %d\n", max_rating_time);
  time_periods = uint32_t( (max_rating_time - time_my_epoch) / float(time_period_length)) + 1;
  printf("+ epoch %d\n", time_my_epoch);
  printf("+ number of time_periods %d\n", time_periods);

  max_train_time_period = 0;

  ostringstream sa;
  sa << "n" << n << "-";
  sa << "m" << m << "-";
  sa << "k" << k;
  if (label != "")
    sa << "-" << label;
  else if (datfname.length() > 3) {
    string q = datfname.substr(0,2);
    if (isalpha(q[0]))
      sa << "-" << q;
  }

  //if (a != 0.3)
  //  sa << "-a" << a;

  //if (b != 0.3)
  //  sa << "-b" << b;

  //if (c != 0.3)
  //  sa << "-c" << c;

  //if (d != 0.3)
  //  sa << "-d" << d;

  if (batch)
    sa << "-batch";

  if (binary_data)
    sa << "-bin";
  
  if (normal_priors)
    sa << "-normpriors";

  if (dynamic_item_representations)
    sa << "-dynitemrep";

  if (dynamic_user_and_item_representations)
    sa << "-dui";

  if (num_threads > 1)
    sa << "-nthreads" << num_threads; 

  if (fixed_item_param)
    sa << "-fip";

  if (pf_init)
    sa << "-pf_init";

  if (pf_init_static)
    sa << "-pf_init_static";

  if (normalized_representations)
    sa << "-normrep";

  sa << "-vprior" << vprior;

  if (seed)
    sa << "-seed" << seed;

  sa << "-tpl" << time_period_length; 

  sa << "-correction";

  //if (rating_threshold)
  //sa << "-rthresh" << rating_threshold;
  
  prefix = sa.str();
  level = Logger::TEST;

  fprintf(stdout, "+ Creating directory %s\n", prefix.c_str());
  fflush(stdout);

  assert (Logger::initialize(prefix, "infer.log", 
			     true, level) >= 0);
  _plogf = fopen(file_str("/param.txt").c_str(), "w");
  if (!_plogf)  {
    printf("cannot open param file:%s\n",  strerror(errno));
    exit(-1);
  }

  plog("n", n);
  plog("k", k);
  plog("t", t);
  plog("seed", seed);
  plog("reportfreq", reportfreq);
  plog("normal_priors", normal_priors);
  plog("datfname", datfname); 
  plog("fixed_item_param", fixed_item_param);
  plog("pf_init", pf_init);
  plog("pf_init_static", pf_init_static);
  plog("normalized_representations", normalized_representations);
  plog("vprior", vprior);
  plog("dynamic_item_representations", dynamic_item_representations);
  plog("dynamic_user_and_item_representations", dynamic_user_and_item_representations);
  plog("num_threads", num_threads);
  plog("time_period_length", time_period_length); 
  plog("time_my_epoch", time_my_epoch);
  plog("time_periods", time_periods); 

  //string ndatfname = file_str("/network.dat");
  //unlink(ndatfname.c_str());
  //assert (symlink(datfname.c_str(), ndatfname.c_str()) >= 0);
  //unlink(file_str("/mutual.txt").c_str());
}

/*
   src: http://www.delorie.com/gnu/docs/glibc/libc_428.html
   Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.
*/
inline int
timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

inline void
timeval_add (struct timeval *result, const struct timeval *x)
{
  result->tv_sec  += x->tv_sec;
  result->tv_usec += x->tv_usec;

  if (result->tv_usec >= 1000000) {
    result->tv_sec++;
    result->tv_usec -= 1000000;
  }
}
#endif
