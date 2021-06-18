#ifndef SPIRV_INCLUDE_LIBRARY_H_
#define SPIRV_INCLUDE_LIBRARY_H_
#include <stdint.h>
extern "C" {
int64_t _Z12get_group_idj(unsigned int);
int64_t _Z14get_local_sizej(unsigned int);
int64_t _Z12get_local_idj(unsigned int);
int64_t _Z14get_num_groupsj(unsigned int);
int _Z10atomic_addPU8CLglobalVii(int *, int);
void _Z7barrierj(unsigned int dim);
void wrapper_barrier(unsigned long long);
float __nv_fast_log2f(float);
float __nv_fast_powf(float, float);
void __assertfail(char *, char *, int, char *, unsigned int);

void setup_idx(int64_t global_idx);
void setup_block_size(int64_t _local_size_x, int64_t _local_size_y,
                      int64_t _local_size_z);
void setup_grid_size(int64_t _group_size_x, int64_t _group_size_y,
                     int64_t _group_size_z);
}
extern __thread int64_t thread_id_x;
extern __thread int64_t thread_id_y;
extern __thread int64_t thread_id_z;
extern __thread int64_t group_id_x;
extern __thread int64_t group_id_y;
extern __thread int64_t group_id_z;
extern __thread int64_t global_id;
extern __thread int pingpong_lock;
extern int64_t local_size_x;
extern int64_t local_size_y;
extern int64_t local_size_z;
extern int64_t group_size_x;
extern int64_t group_size_y;
extern int64_t group_size_z;
#endif
