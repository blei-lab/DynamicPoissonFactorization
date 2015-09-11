#ifndef RATINGS_HH
#define RATINGS_HH

#include <string>
#include <vector>
#include <queue>
#include <map>
#include <stdint.h>
#include "matrix.hh"
#include "env.hh"
#include <string.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf.h>

using namespace std;

class Ratings {
public:
  Ratings(Env &env):
    _users2rating(env.n),
    _users(env.n),
    _movies(env.m),
    _env(env),
    _curr_user_seq(0), 
    _curr_movie_seq(0),
    _nratings(0),
    _likes(0) { }
  ~Ratings() { }

  int read(string s);
  uint32_t input_rating_class(uint32_t v) const;
  bool test_hit(uint32_t v) const;
  //int write_marginal_distributions();
  uint32_t time_period(uint32_t rating_time);

  const SparseMatrixT &users() const { return _users; }
  SparseMatrixT &users() { return _users; }

  const SparseMatrixT &movies() const { return _movies; }
  SparseMatrixT &movies() { return _movies; }
  
  uint32_t n() const;
  uint32_t m() const;
  uint32_t r(uint32_t i, uint32_t j, uint32_t t) const;
  uint32_t allr(uint32_t a, uint32_t b) const;
  uint32_t allr(uint32_t a, uint32_t b, uint32_t &t) const;
  uint32_t last_user_time_period(uint32_t n);

  uint32_t nratings() const { return _nratings; }
  uint32_t likes() const { return _likes; }
  const vector<Rating> &allratings() const { return _ratings; }
 
  const vector<uint32_t> *get_users(uint32_t a, uint32_t t);
  const vector<uint32_t> *get_movies(uint32_t a, uint32_t t);

  const IDMap &user2seq() const { return _user2seq; }
  const IDMap &seq2user() const { return _seq2user; }

  const IDMap &movie2seq() const { return _movie2seq; }
  const IDMap &seq2movie() const { return _seq2movie; }
  int read_generic(FILE *f, CountMap *m);
  
  int read_test_users(FILE *f, UserMap *);

  string movie_type(uint32_t movie_seq) const;
  string movie_name(uint32_t movie_seq) const;
  IDMap _movie2birthtimestep;
  
private:
  int read_generic_train(string dir);
  //string movies_by_user_s() const;
  bool add_movie(uint32_t id);
  bool add_user(uint32_t id);

  SparseMatrixTR _users2rating; // user-indexed, then time-indexed, then movie-indexed -> rating 
  SparseMatrixT _users;
  SparseMatrixT _movies;
  vector<Rating> _ratings;

  Env &_env;
  IDMap _user2seq;
  IDMap _movie2seq;
  IDMap _seq2user;
  IDMap _seq2movie;
  uint32_t _curr_user_seq;
  uint32_t _curr_movie_seq;
  uint32_t _nratings;
  uint32_t _likes;
  StrMapInv _movie_names;
  StrMapInv _movie_types;
};

inline uint32_t
Ratings::n() const
{
  return _users.size();
}

inline uint32_t
Ratings::m() const
{
  return _movies.size();
}

inline bool
Ratings::add_user(uint32_t id)
{
  if (_curr_user_seq >= _env.n) {
    debug("max users %d reached", _env.n);
    return false;
  }
  _user2seq[id] = _curr_user_seq;
  _seq2user[_curr_user_seq] = id;

  //assert (!_users[_curr_user_seq]);
  std::unordered_map<uint32_t, std::vector<uint32_t> *> **v = _users.data();
  v[_curr_user_seq] = new std::unordered_map<uint32_t, std::vector<uint32_t> *>; 
  for(int t=0; t<_env.time_periods; ++t)
      (*v[_curr_user_seq])[t] = new vector<uint32_t>;
  map<uint32_t, RatingMap *> **u2r = _users2rating.data();
  //rm[_curr_user_seq] = new RatingMap;
  u2r[_curr_user_seq] = new map<uint32_t, RatingMap *>;

  // map<uint32_t, RatingMap *> *v = _users2rating[n];

  // TODO: if we go sparse then this isn't necessary for every time slice
  for(uint32_t t=0; t<_env.time_periods; ++t)
      (*u2r[_curr_user_seq])[t] = new RatingMap; 


  _curr_user_seq++;
  return true;
}

inline bool
Ratings::add_movie(uint32_t id)
{
  if (_curr_movie_seq >= _env.m) {
    debug("max movies %d reached", _env.m);
    return false;
  }
  _movie2seq[id] = _curr_movie_seq;
  _seq2movie[_curr_movie_seq] = id;

  //assert (!_movies[_curr_movie_seq]);

  std::unordered_map<uint32_t, std::vector<uint32_t> *> **v = _movies.data();
  v[_curr_movie_seq] = new std::unordered_map<uint32_t, std::vector<uint32_t> *>; 
  for(uint32_t t=0; t<_env.time_periods; ++t)
      (*v[_curr_movie_seq])[t] = new vector<uint32_t>;
  _curr_movie_seq++;
  return true;
}

inline uint32_t
Ratings::allr(uint32_t a, uint32_t b) const
{
    uint32_t r; 

   for(uint32_t t=0; t<_env.time_periods; ++t) {
        r = this->r(a, b, t);
        if(r != 0)
            return r;
    }
    return 0;
}

inline uint32_t
Ratings::allr(uint32_t a, uint32_t b, uint32_t &ts) const
{
    uint32_t r; 

   for(uint32_t t=0; t<_env.time_periods; ++t) {
        r = this->r(a, b, t);
        if(r != 0) {
            ts=t;
            return r;
        }
    }
    return 0;
}

inline uint32_t
Ratings::r(uint32_t a, uint32_t b, uint32_t t) const
{
  assert (a < _env.n && b < _env.m);
  const RatingMap *rm = (*_users2rating[a])[t];
  assert(rm);
  const RatingMap &rmc = *rm;
  RatingMap::const_iterator itr = rmc.find(b);
  if (itr == rmc.end())
    return 0;
  else
    return itr->second;
}

inline const vector<uint32_t> *
Ratings::get_users(uint32_t a, uint32_t t)
{
  assert (a < _movies.size());
  const vector<uint32_t> *v = (*_movies[a])[t];
  return v;
}

inline const vector<uint32_t> *
Ratings::get_movies(uint32_t a, uint32_t t)
{
  assert (a < _users.size());
  const vector<uint32_t> *v = (*_users[a])[t];
  return v;
}

inline bool
Ratings::test_hit(uint32_t v) const
{
  if (_env.binary_data)
    return v >= 1;
  return v >= _env.rating_threshold;
}

inline uint32_t
Ratings::input_rating_class(uint32_t v) const
{
  if (!_env.binary_data)
    return v;
  return v >= _env.rating_threshold  ? 1 : 0;
}

inline string
Ratings::movie_name(uint32_t movie_seq) const
{
  assert (movie_seq < _env.m);
  StrMapInv::const_iterator i = _movie_names.find(movie_seq);
  if (i != _movie_names.end())
    return i->second;
  return "";
}

inline string
Ratings::movie_type(uint32_t movie_seq) const
{
  assert (movie_seq < _env.m);
  StrMapInv::const_iterator i = _movie_types.find(movie_seq);
  if (i != _movie_types.end())
    return i->second;
  return "";
}

#endif
