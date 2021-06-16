#include "library.h" // built-in function
#include <cstdio>
#include <pthread.h>

#include "../common.h" // (in the main program folder)	needed to recognized input parameters
#include "../util/timer/timer.h" // (in library path specified to compiler)	needed by timer
#include "./kernel_gpu_cuda_wrapper_2.h" // (in the current directory)

extern "C" {
void *_Z10findRangeKlP5knodelPlS1_S1_S1_PiS2_S2_S2__wrapper(void *);
}

void *wrapper_func_2(void *p) {
  int **ret = (int **)p;
  int tid = *(ret[0]);
  setup_idx(tid);
  _Z10findRangeKlP5knodelPlS1_S1_S1_PiS2_S2_S2__wrapper((void *)(ret + 1));
}

void *gen_input_2(int tid, long height, knode *knodesD, long knodes_elem,
                  long *currKnodeD, long *offsetD, long *lastKnodeD,
                  long *offset_2D, int *startD, int *endD, int *RecstartD,
                  int *ReclenD) {
  int **ret = new int *[12];

  int *p0 = new int;
  *p0 = tid;
  ret[0] = (int *)p0;

  long *p1 = new long;
  *p1 = height;
  ret[1] = (int *)p1;

  knode **p2 = (knode **)malloc(sizeof(knode *));
  *p2 = knodesD;
  ret[2] = (int *)(p2);

  long *p3 = new long;
  *p3 = knodes_elem;
  ret[3] = (int *)p3;

  long **p4 = (long **)malloc(sizeof(long *));
  *p4 = currKnodeD;
  ret[4] = (int *)(p4);

  long **p5 = (long **)malloc(sizeof(long *));
  *p5 = offsetD;
  ret[5] = (int *)(p5);

  long **p6 = (long **)malloc(sizeof(long *));
  *p6 = lastKnodeD;
  ret[6] = (int *)(p6);

  long **p7 = (long **)malloc(sizeof(long *));
  *p7 = offset_2D;
  ret[7] = (int *)(p7);

  int **p8 = (int **)malloc(sizeof(int *));
  *p8 = startD;
  ret[8] = (int *)(p8);

  int **p9 = (int **)malloc(sizeof(int *));
  *p9 = endD;
  ret[9] = (int *)(p9);

  int **p10 = (int **)malloc(sizeof(int *));
  *p10 = RecstartD;
  ret[10] = (int *)(p10);

  int **p11 = (int **)malloc(sizeof(int *));
  *p11 = ReclenD;
  ret[11] = (int *)(p11);
  return (void *)ret;
}

void kernel_gpu_cuda_wrapper_2(knode *knodes, long knodes_elem, long knodes_mem,
                               int order, long maxheight, int count,
                               long *currKnode, long *offset, long *lastKnode,
                               long *offset_2, int *start, int *end,
                               int *recstart, int *reclength) {
  // timer
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;

  time0 = get_time();

  //====================================================================================================100
  //	EXECUTION PARAMETERS
  //====================================================================================================100

  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 1024 ? order : 1024;

  printf("# of blocks = %d, # of threads/block = %d (ensure that device can "
         "handle)\n",
         numBlocks, threadsPerBlock);

  time1 = get_time();
  time2 = get_time();
  time3 = get_time();

  //======================================================================================================================================================150
  //	KERNEL
  //======================================================================================================================================================150
  int NUM_THREADS = numBlocks * threadsPerBlock;
  pthread_t *threads = new pthread_t[NUM_THREADS];

  int rc;
  int *thread_id = new int[NUM_THREADS];

  // set grid, block dim
  setup_grid_size(numBlocks, 1, 1);
  setup_block_size(threadsPerBlock, 1, 1);

  for (long t = 0; t < NUM_THREADS; t++) {
    void *inp =
        gen_input_2(t, maxheight, knodes, knodes_elem, currKnode, offset,
                    lastKnode, offset_2, start, end, recstart, reclength);
    rc = pthread_create(&threads[t], NULL, wrapper_func_2, inp);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }
  /* Last thing that main() should do */
  for (long t = 0; t < NUM_THREADS; t++)
    pthread_join(threads[t], NULL);

  time4 = get_time();
  time5 = get_time();
  time6 = get_time();

  //======================================================================================================================================================150
  //	DISPLAY TIMING
  //======================================================================================================================================================150

  printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

  printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",
         (float)(time1 - time0) / 1000000,
         (float)(time1 - time0) / (float)(time6 - time0) * 100);
  printf("%15.12f s, %15.12f % : GPU MEM: ALO\n",
         (float)(time2 - time1) / 1000000,
         (float)(time2 - time1) / (float)(time6 - time0) * 100);
  printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",
         (float)(time3 - time2) / 1000000,
         (float)(time3 - time2) / (float)(time6 - time0) * 100);

  printf("%15.12f s, %15.12f % : GPU: KERNEL\n",
         (float)(time4 - time3) / 1000000,
         (float)(time4 - time3) / (float)(time6 - time0) * 100);

  printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",
         (float)(time5 - time4) / 1000000,
         (float)(time5 - time4) / (float)(time6 - time0) * 100);
  printf("%15.12f s, %15.12f % : GPU MEM: FRE\n",
         (float)(time6 - time5) / 1000000,
         (float)(time6 - time5) / (float)(time6 - time0) * 100);

  printf("Total time:\n");
  printf("%.12f s\n", (float)(time6 - time0) / 1000000);
}
