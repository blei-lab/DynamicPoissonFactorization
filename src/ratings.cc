#include "ratings.hh"
#include "log.hh"
#include <wchar.h>

int
Ratings::read(string s)
{
  fprintf(stdout, "+ reading ratings dataset from %s\n", s.c_str());
  fflush(stdout);

  read_generic_train(s);
  //write_marginal_distributions();
    
  char st[1024];
  sprintf(st, "read %d users, %d movies, %d ratings", 
	  _curr_user_seq, _curr_movie_seq, _nratings);
  _env.n = _curr_user_seq;
  _env.m = _curr_movie_seq;
  Env::plog("statistics", string(st));

  return 0;
}

int
Ratings::read_generic_train(string dir)
{
  char buf[1024];
  sprintf(buf, "%s/train.tsv", dir.c_str());
  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }

  read_generic(f, NULL);
  Env::plog("training ratings", _nratings);

  uint32_t t = 0;
  for(uint32_t i=0; i<_env.n; ++i) {
    t = last_user_time_period(i);
    if (t > _env.max_train_time_period)
      _env.max_train_time_period = t;
  }

  fclose(f);
  lerr("training timesteps %d\n", _env.max_train_time_period);

  return 0;

}

uint32_t 
Ratings::time_period(uint32_t rating_time)
{
  uint32_t rating_time_period; 
  rating_time_period = (rating_time - _env.time_my_epoch) / _env.time_period_length; 
  // Env::plog("returning time period %d\n", rating_time_period);
  if (rating_time_period >= _env.time_periods) {
    printf("%d - %d\n", rating_time, _env.time_my_epoch);
    printf("current time period >= than number of time_periods: %d > %d", rating_time_period, _env.time_periods); 
    exit(-1); 
  }
  return rating_time_period; 
}

uint32_t 
Ratings::last_user_time_period(uint32_t n)
{
        map<uint32_t, RatingMap *> *v = _users2rating[n];

        if(!v) // is this correct? 
            return 0;

        uint32_t maxt=0; 
        for (uint32_t t=0; t<_env.time_periods; ++t) {
              if ((*v)[t]->size() > 0)
                maxt = t; 
        }
        return maxt;
}

int
Ratings::read_generic(FILE *f, CountMap *cmap)
{
  assert(f);
  char b[128];
  uint32_t mid = 0, uid = 0, rating = 0, rating_time = 0;
  while (!feof(f)) {
    if (fscanf(f, "%u\t%u\t%u\t%u\n", &uid, &mid, &rating, &rating_time) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    if (_env.dynamic_item_representations) {
        uint32_t tmp; 
        tmp = mid;
        mid = uid;
        uid = tmp;
    }

    uint32_t rating_time_period = time_period(rating_time); 

    IDMap::iterator it = _user2seq.find(uid);
    IDMap::iterator mt = _movie2seq.find(mid);

    if ((it == _user2seq.end() && _curr_user_seq >= _env.n) ||
        (mt == _movie2seq.end() && _curr_movie_seq >= _env.m))
      continue;

    if (input_rating_class(rating) == 0)
      continue;

    if (it == _user2seq.end())
      assert(add_user(uid));

    if (mt == _movie2seq.end())
      assert(add_movie(mid));

    uint32_t m = _movie2seq[mid];
    uint32_t n = _user2seq[uid];

    if (input_rating_class(rating) > 0) {
      if (!cmap) {
        lerr("adding train entry for user %d, item %d, time %d\n", n, m, rating_time_period);
        _nratings++;

        map<uint32_t, RatingMap *> *v = _users2rating[n];

        //if (v->find(rating_time_period) == v->end())
        //  (*v)[rating_time_period] = new RatingMap; 

        if (_env.binary_data)
          (*((*v)[rating_time_period]))[m] = 1;
        else {
          assert (rating > 0);
          (*((*v)[rating_time_period]))[m] = rating;
        }
        (*_users[n])[rating_time_period]->push_back(m);
        (*_movies[m])[rating_time_period]->push_back(n);

        if (_movie2birthtimestep.find(m) == _user2seq.end() || _movie2birthtimestep[m] > rating_time_period)
            _movie2birthtimestep[m] = rating_time_period;

      } else {
        lerr("adding test or validation entry for user %d, item %d, time %d\n", n, m, rating_time_period);
        Rating r(n,m,rating_time_period);
        assert(cmap);
        if (_env.binary_data)
          (*cmap)[r] = 1;
        else
          (*cmap)[r] = rating;
      }
    }
  }
  return 0;
}

#if 0 
int
Ratings::write_marginal_distributions()
{
  FILE *f = fopen(Env::file_str("/byusers.tsv").c_str(), "w");
  uint32_t x = 0;
  uint32_t nusers = 0;
  for (uint32_t t = 0; t < _env.max_time_period; ++t) { 
    for (uint32_t n = 0; n < _env.n; ++n) {
      const vector<uint32_t> *movies = get_movies(n);
      IDMap::const_iterator it = seq2user().find(n);
      if (!movies || movies->size() == 0) {
        debug("0 movies for user %d (%d)", n, it->second);
        x++;
        continue;
      }
      uint32_t t = 0;
      for (uint32_t m = 0; m < movies->size(); m++) {
        uint32_t mov = (*movies)[m];
        yval_t y = r(n,mov);
        t += y;
      }
      x = 0;
      fprintf(f, "%d\t%d\t%d\t%d\n", n, it->second, movies->size(), t);
      nusers++;
    }
  }
  fclose(f);
  //_env.n = nusers;
  lerr("longest sequence of users with no movies: %d", x);

  f = fopen(Env::file_str("/byitems.tsv").c_str(), "w");
  x = 0;
  uint32_t nitems = 0;
  for (uint32_t n = 0; n < _env.m; ++n) {
    const vector<uint32_t> *users = get_users(n);
    IDMap::const_iterator it = seq2movie().find(n);
    if (!users || users->size() == 0) {
      lerr("0 users for movie %d (%d)", n, it->second);
      x++;
      continue;
    }
    uint32_t t = 0;
    for (uint32_t m = 0; m < users->size(); m++) {
      uint32_t u = (*users)[m];
      yval_t y = r(u,n);
      t += y;
    }
    x = 0;
    fprintf(f, "%d\t%d\t%d\t%d\n", n, it->second, users->size(), t);
    nitems++;
  }
  fclose(f);
  //_env.m = nitems;
  lerr("longest sequence of items with no users: %d", x);
  Env::plog("post pruning nusers:", _env.n);
  Env::plog("post pruning nitems:", _env.m);
}
#endif

int
Ratings::read_test_users(FILE *f, UserMap *bmap)
{
  assert (bmap);
  uint32_t uid = 0;
  while (!feof(f)) {
    if (fscanf(f, "%u\n", &uid) < 0) {
      printf("error: unexpected lines in file\n");
      exit(-1);
    }

    if(_env.dynamic_item_representations) {
        IDMap::iterator it = _movie2seq.find(uid);
        if (it == _movie2seq.end())
          continue;
        uint32_t n = _movie2seq[uid];
        (*bmap)[n] = true;
    } else {
        IDMap::iterator it = _user2seq.find(uid);
        if (it == _user2seq.end())
          continue;
        uint32_t n = _user2seq[uid];
        (*bmap)[n] = true;
    }
  }
  Env::plog("read %d test users", bmap->size());
  return 0;
}

#if 0
string
Ratings::movies_by_user_s() const
{
  ostringstream sa;
  sa << "\n[\n";
  for (uint32_t i = 0; i < _users.size(); ++i) {
    IDMap::const_iterator it = _seq2user.find(i);
    sa << it->second << ":";
    vector<uint32_t> *v = _users[i];
    if (v)  {
      for (uint32_t j = 0; j < v->size(); ++j) {
	uint32_t m = v->at(j);
	IDMap::const_iterator mt = _seq2movie.find(m);
	sa << mt->second;
	if (j < v->size() - 1)
	  sa << ", ";
      }
      sa << "\n";
    }
  }
  sa << "]";
  return sa.str();
}
#endif
