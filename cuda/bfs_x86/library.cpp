#include "library.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <pthread.h>
#define BLOCK_DIM_X local_size_x
#define BLOCK_DIM_Y local_size_y
#define BLOCK_DIM_Z local_size_z
__thread int64_t thread_id_x = 1;
__thread int64_t thread_id_y = 1;
__thread int64_t thread_id_z = 1;
__thread int64_t group_id_x = 1;
__thread int64_t group_id_y = 1;
__thread int64_t group_id_z = 1;
__thread int64_t global_id = 1;
int64_t local_size_x = 1;
int64_t local_size_y = 1;
int64_t local_size_z = 1;
int64_t group_size_x = 1;
int64_t group_size_y = 1;
int64_t group_size_z = 1;
const int MAX_BLOCK_CNT = 1000;
// TODO: 10000 is the maximum group number
// using double buffer for ping-pong
int *_sync_counter = new int[MAX_BLOCK_CNT * 2];
pthread_cond_t g_cond[MAX_BLOCK_CNT];
pthread_mutex_t lock;

void setup_block_size(int64_t _local_size_x, int64_t _local_size_y,
                      int64_t _local_size_z) {
  local_size_x = _local_size_x;
  local_size_y = _local_size_y;
  local_size_z = _local_size_z;
}
void setup_grid_size(int64_t _group_size_x, int64_t _group_size_y,
                     int64_t _group_size_z) {
  if (_group_size_x * _group_size_y * _group_size_z > MAX_BLOCK_CNT) {
    printf("we can not support too many blocks\n");
    exit(1);
  }
  group_size_x = _group_size_x;
  group_size_y = _group_size_y;
  group_size_z = _group_size_z;
}

void setup_idx(int64_t global_idx) {
  global_id = global_idx;
  // set block index
  int64_t block_idx = global_idx / (local_size_x * local_size_y * local_size_z);
  group_id_x = block_idx % group_size_x;
  block_idx /= group_size_x;
  group_id_y = block_idx % group_size_y;
  group_id_z = block_idx / group_size_y;
  // set thread index
  int64_t thread_idx =
      global_idx % (local_size_x * local_size_y * local_size_z);
  thread_id_x = thread_idx % local_size_x;
  thread_idx /= local_size_x;
  thread_id_y = thread_idx % local_size_y;
  thread_id_z = thread_idx / local_size_y;
}

int64_t _Z12get_group_idj(unsigned int dim) {
  switch (dim) {
  case 0:
    return group_id_x;
  case 1:
    return group_id_y;
  case 2:
    return group_id_z;
  default:
    printf("Error: only support 3-dim grid\n");
    exit(1);
  }
}
int64_t _Z14get_local_sizej(unsigned int dim) {
  switch (dim) {
  case 0:
    return local_size_x;
  case 1:
    return local_size_y;
  case 2:
    return local_size_z;
  default:
    printf("Error: only support 3-dim\n");
    exit(1);
  }
  return 1;
}
int64_t _Z12get_local_idj(unsigned int dim) {
  switch (dim) {
  case 0:
    return thread_id_x;
  case 1:
    return thread_id_y;
  case 2:
    return thread_id_z;
  default:
    printf("Error: only support 3-dim block\n");
    exit(1);
  }
  return 1;
}

int64_t _Z14get_num_groupsj(unsigned int dim) {
  switch (dim) {
  case 0:
    return group_size_x;
  case 1:
    return group_size_y;
  case 2:
    return group_size_z;
  default:
    printf("Error: only support 3-dim block\n");
    exit(1);
  }
  return 1;
}

void _Z7barrierj(unsigned int dim) { wrapper_barrier(global_id); }

void wrapper_barrier(unsigned long long global_idx) {
  // first calculate group idx
  unsigned long long group_idx =
      global_idx / (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z);
  pthread_mutex_lock(&lock);
  _sync_counter[group_idx]++;
  if (_sync_counter[group_idx] % (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z))
    pthread_cond_wait(&g_cond[group_idx], &lock);
  else {
    _sync_counter[group_idx] = 0;
    pthread_cond_broadcast(&g_cond[group_idx]);
  }
  pthread_mutex_unlock(&lock);
}

void __assertfail(char *msg, char *file, int line, char *function,
                  unsigned int datatype) {
  printf("Assert Error: %s in %s:%d func: %s\n", msg, file, line, function);
  exit(1);
}

int _Z10atomic_addPU8CLglobalVii(int *p, int val) {
  pthread_mutex_lock(&lock);
  int old = *p;
  *p = (old + val);
  pthread_mutex_unlock(&lock);
  return old;
}

float __nv_fast_log2f(float x) { return log2f(x); }
float __nv_fast_powf(float base, float exponent) {
  return powf(base, exponent);
}