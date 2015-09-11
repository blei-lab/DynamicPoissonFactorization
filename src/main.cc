#include "env.hh"
#include "dynnormprec.hh"
#include "ratings.hh"

#include <stdlib.h>
#include <string>
#include <sstream>
#include <signal.h>

string Env::prefix = "";
Logger::Level Env::level = Logger::DEBUG;
FILE *Env::_plogf = NULL;
void usage();
void test();

Env *env_global = NULL;
volatile sig_atomic_t sig_handler_active = 0;

void
term_handler(int sig)
{
  if (env_global) {
    printf("Got signal. Saving model state.\n");
    fflush(stdout);
    env_global->save_state_now = 1;
  } else {
    signal(sig, SIG_DFL);
    raise(sig);
  }
}

int
main(int argc, char **argv)
{
  signal(SIGTERM, term_handler);
  if (argc <= 1) {
    printf("dynnormprec -dir <netflix-dataset-dir> -n <users>" \
	   "-m <movies> -k <dims> \n");
    exit(0);
  }

  string fname;
  uint32_t n = 0, m = 0;
  uint32_t k = 0;
  uint32_t rfreq = 10;
  uint32_t max_iterations = 1000;
  double rand_seed = 0;

  bool test = false;
  bool batch = true;

  bool model_load = false;
  string model_location = "";

  bool binary_data = true;
  bool gen_ranking_for_users = false;
  uint32_t rating_threshold = 1;
  double vprior = 1.0; 

  bool normal_priors = true; 

  bool dynamic_item_representations = false; 
  bool dynamic_ui = true; 

  bool fixed_item_param = false;
  bool pf_init = false; 
  bool pf_init_static = false;

  uint32_t num_threads = 1;

  int one_month = 60*60*24*30;
  uint32_t time_period_length = 6*one_month;

  uint32_t i = 0;
  while (i <= argc - 1) {
    if (strcmp(argv[i], "-dir") == 0) {
      fname = string(argv[++i]);
      fprintf(stdout, "+ dir = %s\n", fname.c_str());
    } else if (strcmp(argv[i], "-n") == 0) {
      n = atoi(argv[++i]);
      fprintf(stdout, "+ n = %d\n", n);
    } else if (strcmp(argv[i], "-m") == 0) {
      m = atoi(argv[++i]);
      fprintf(stdout, "+ m = %d\n", m);
    } else if (strcmp(argv[i], "-k") == 0) {
      k = atoi(argv[++i]);
      fprintf(stdout, "+ k = %d\n", k);
    } else if (strcmp(argv[i], "-rfreq") == 0) {
      rfreq = atoi(argv[++i]);
      fprintf(stdout, "+ rfreq = %d\n", rfreq);
    } else if (strcmp(argv[i], "-max-iterations") == 0) {
      max_iterations = atoi(argv[++i]);
      fprintf(stdout, "+ max iterations %d\n", max_iterations);
    } else if (strcmp(argv[i], "-seed") == 0) {
      rand_seed = atof(argv[++i]);
      fprintf(stdout, "+ random seed set to %.5f\n", rand_seed);
    } else if (strcmp(argv[i], "-load") == 0) {
      model_load = true;
      model_location = string(argv[++i]);
      fprintf(stdout, "+ loading theta from %s\n", model_location.c_str());
    } else if (strcmp(argv[i], "-test") == 0) {
      test = true;
      fprintf(stdout, "+ test mode\n");
    } else if (strcmp(argv[i], "-batch") == 0) {
      batch = true;
      fprintf(stdout, "+ batch inference\n");
    } else if (strcmp(argv[i], "-fixed-item-param") == 0) { 
       fixed_item_param = true;  
    } else if (strcmp(argv[i], "-pf-init") == 0) { 
       pf_init = true;  
    } else if (strcmp(argv[i], "-pf-init-static") == 0) {
       pf_init_static = true;
    } else if (strcmp(argv[i], "-non-binary-data") == 0) {
      binary_data = false;
    } else if (strcmp(argv[i], "-gen-ranking") == 0) {
      gen_ranking_for_users = true;
    } else if (strcmp(argv[i], "-rating-threshold") == 0) {
      rating_threshold = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-normal-priors") == 0) {
      normal_priors = true;
    } else if (strcmp(argv[i], "-vprior") == 0) {
      vprior = atof(argv[++i]);
    } else if (strcmp(argv[i], "-dynamic_item_representations") == 0) {
      dynamic_item_representations = true;
    } else if (strcmp(argv[i], "-dynamic_ui") == 0) {
      dynamic_ui = true;
    } else if (strcmp(argv[i], "-num_threads") == 0) {
      num_threads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-tpl") == 0) {
      time_period_length = atoi(argv[++i]);
    } else if (i > 0) {
      fprintf(stdout,  "error: unknown option %s\n", argv[i]);
      assert(0);
    } 
    ++i;
  };

  if(dynamic_item_representations) {
    uint32_t tmp = n;
    n = m;
    m = tmp;
  }

  Env env(n, m, k, fname, rfreq, 
	  rand_seed, max_iterations, 
	  model_load, model_location, 
	  batch, binary_data, 
	  rating_threshold, 
	  normal_priors, fixed_item_param, pf_init, pf_init_static,
      vprior, dynamic_item_representations,
      dynamic_ui, num_threads, time_period_length);
  env_global = &env;

  Ratings ratings(env);
  if (ratings.read(fname.c_str()) < 0) {
    fprintf(stderr, "error reading dataset from dir %s; quitting\n", 
	    fname.c_str());
    return -1;
  }

  if (gen_ranking_for_users) {
    DynNormPRec dynnormprec(env, ratings);
    dynnormprec.gen_ranking_for_users(true);
    exit(0);
  }


  if (normal_priors) { 
    DynNormPRec dynnormprec(env, ratings); 

    if (env.dynamic_user_and_item_representations)
      dynnormprec.infer_dui_correction();
    else
      dynnormprec.infer(); 
    exit(0); 
  } 

}
