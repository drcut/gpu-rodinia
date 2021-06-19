/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.
 **   The sequential version is gaussian.c.  This parallel
 **   implementation converts three independent for() loops
 **   into three Fans.  Use the data file ge_3.dat to verify
 **   the correction of the output.
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */
#include "library.h" // built-in function
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifdef TIMING
#include "timing.h"
#endif

#ifdef RD_WG_SIZE_0_0
#define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define MAXBLOCKSIZE RD_WG_SIZE
#else
#define MAXBLOCKSIZE 512
#endif

// 2D defines. Go from specific to general
#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_XY RD_WG_SIZE
#else
#define BLOCK_SIZE_XY 4
#endif

#ifdef TIMING
struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

int Size;
float *a, *b, *finalVec;
float *m;

FILE *fp;

void InitProblemOnce(char *filename);
void InitPerRun();
void ForwardSub();
void BackSub();
extern "C" {
void *_Z4Fan1PfS_ii_wrapper(void *);
void *_Z4Fan2PfS_S_iii_wrapper(void *);
}
void *wrapper_Fan1(void *p) {
  int **ret = (int **)p;
  int tid = *(ret[0]);
  setup_idx(tid);
  _Z4Fan1PfS_ii_wrapper((void *)(ret + 1));
  return NULL;
}
void *wrapper_Fan2(void *p) {
  int **ret = (int **)p;
  int tid = *(ret[0]);
  setup_idx(tid);
  _Z4Fan2PfS_S_iii_wrapper((void *)(ret + 1));
  return NULL;
}

void *gen_input_Fan1(int tid, float *m_cuda, float *a_cuda, int Size, int t) {
  int **ret = new int *[5];

  int *p0 = new int;
  *p0 = tid;
  ret[0] = (int *)p0;

  float **p1 = new float *;
  *p1 = m_cuda;
  ret[1] = (int *)(p1);

  float **p2 = new float *;
  *p2 = a_cuda;
  ret[2] = (int *)(p2);

  int *p3 = new int;
  *p3 = Size;
  ret[3] = (int *)p3;

  int *p4 = new int;
  *p4 = t;
  ret[4] = (int *)p4;

  return (void *)ret;
}

void *gen_input_Fan2(int tid, float *m_cuda, float *a_cuda, float *b_cuda,
                     int Size, int j1, int t) {
  int **ret = new int *[7];

  int *p0 = new int;
  *p0 = tid;
  ret[0] = (int *)p0;

  float **p1 = new float *;
  *p1 = m_cuda;
  ret[1] = (int *)(p1);

  float **p2 = new float *;
  *p2 = a_cuda;
  ret[2] = (int *)(p2);

  float **p3 = new float *;
  *p3 = b_cuda;
  ret[3] = (int *)(p3);

  int *p4 = new int;
  *p4 = Size;
  ret[4] = (int *)p4;

  int *p5 = new int;
  *p5 = j1;
  ret[5] = (int *)p5;

  int *p6 = new int;
  *p6 = t;
  ret[6] = (int *)p6;

  return (void *)ret;
}

void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);

unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void create_matrix(float *m, int size) {
  int i, j;
  float lamda = -0.01;
  float coe[2 * size - 1];
  float coe_i = 0.0;

  for (i = 0; i < size; i++) {
    coe_i = 10 * exp(lamda * i);
    j = size - 1 + i;
    coe[j] = coe_i;
    j = size - 1 - i;
    coe[j] = coe_i;
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      m[i * size + j] = coe[size - 1 - i + j];
    }
  }
}

inline int max(int l, int r) { return (l > r) ? l : r; }
int main(int argc, char *argv[]) {
  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n",
         MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
  int verbose = 1;
  int i, j;
  char flag;
  if (argc < 2) {
    printf("Usage: gaussian -f filename / -s size [-q]\n\n");
    printf("-q (quiet) suppresses printing the matrix and result values.\n");
    printf("-f (filename) path of input file\n");
    printf(
        "-s (size) size of matrix. Create matrix and rhs in this program \n");
    printf(
        "The first line of the file contains the dimension of the matrix, n.");
    printf("The second line of the file is a newline.\n");
    printf("The next n lines contain n tab separated values for the matrix.");
    printf("The next line of the file is a newline.\n");
    printf("The next line of the file is a 1xn vector with tab separated "
           "values.\n");
    printf("The next line of the file is a newline. (optional)\n");
    printf("The final line of the file is the pre-computed solution. "
           "(optional)\n");
    printf("Example: matrix4.txt:\n");
    printf("4\n");
    printf("\n");
    printf("-0.6	-0.5	0.7	0.3\n");
    printf("-0.3	-0.9	0.3	0.7\n");
    printf("-0.4	-0.5	-0.3	-0.8\n");
    printf("0.0	-0.1	0.2	0.9\n");
    printf("\n");
    printf("-0.85	-0.68	0.24	-0.53\n");
    printf("\n");
    printf("0.7	0.0	-0.4	-0.5\n");
    exit(0);
  }

  // char filename[100];
  // sprintf(filename,"matrices/matrix%d.txt",size);

  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-') { // flag
      flag = argv[i][1];
      switch (flag) {
      case 's': // platform
        i++;
        Size = atoi(argv[i]);
        printf("Create matrix internally in parse, size = %d \n", Size);

        a = (float *)malloc(Size * Size * sizeof(float));
        create_matrix(a, Size);

        b = (float *)malloc(Size * sizeof(float));
        for (j = 0; j < Size; j++)
          b[j] = 1.0;

        m = (float *)malloc(Size * Size * sizeof(float));
        break;
      case 'f': // platform
        i++;
        printf("Read file from %s \n", argv[i]);
        InitProblemOnce(argv[i]);
        break;
      case 'q': // quiet
        verbose = 0;
        break;
      }
    }
  }

  // InitProblemOnce(filename);
  InitPerRun();
  // begin timing
  struct timeval time_start;
  gettimeofday(&time_start, NULL);

  // run kernels
  ForwardSub();

  // end timing
  struct timeval time_end;
  gettimeofday(&time_end, NULL);
  unsigned int time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec) -
                            (time_start.tv_sec * 1000000 + time_start.tv_usec);

  if (verbose) {
    printf("Matrix m is: \n");
    PrintMat(m, Size, Size);

    printf("Matrix a is: \n");
    PrintMat(a, Size, Size);

    printf("Array b is: \n");
    PrintAry(b, Size);
  }
  BackSub();
  if (verbose) {
    printf("The final solution is: \n");
    PrintAry(finalVec, Size);
  }
  printf("\nTime total (including memory transfers)\t%f sec\n",
         time_total * 1e-6);
  printf("Time for CUDA kernels:\t%f sec\n", totalKernelTime * 1e-6);

  /*printf("%d,%d\n",size,time_total);
  fprintf(stderr,"%d,%d\n",size,time_total);*/

  free(m);
  free(a);
  free(b);

#ifdef TIMING
  printf("Exec: %f\n", kernel_time);
#endif
}

/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
void InitProblemOnce(char *filename) {
  // char *filename = argv[1];

  // printf("Enter the data file name: ");
  // scanf("%s", filename);
  // printf("The file name is: %s\n", filename);

  fp = fopen(filename, "r");

  fscanf(fp, "%d", &Size);

  a = (float *)malloc(Size * Size * sizeof(float));

  InitMat(a, Size, Size);
  // printf("The input matrix a is:\n");
  // PrintMat(a, Size, Size);
  b = (float *)malloc(Size * sizeof(float));

  InitAry(b, Size);
  // printf("The input array b is:\n");
  // PrintAry(b, Size);

  m = (float *)malloc(Size * Size * sizeof(float));
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() {
  int i;
  for (i = 0; i < Size * Size; i++)
    *(m + i) = 0.0;
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub() {
  int t;
  /*
  float *m_cuda, *a_cuda, *b_cuda;
  */
  int block_size, grid_size;
  block_size = MAXBLOCKSIZE;
  grid_size = (Size / block_size) + (!(Size % block_size) ? 0 : 1);

  int blockSize2d, gridSize2d;
  blockSize2d = BLOCK_SIZE_XY;
  gridSize2d = (Size / blockSize2d) + (!(Size % blockSize2d ? 0 : 1));

  // pthread_t *threads =
  //     new pthread_t[max(block_size * grid_size,
  //                       blockSize2d * blockSize2d * gridSize2d *
  //                       gridSize2d)];
  int rc;

#ifdef TIMING
  gettimeofday(&tv_kernel_start, NULL);
#endif
  int NUM_THREADS_Fan1 = block_size * grid_size;
  pthread_t *threads_Fan1 = new pthread_t[NUM_THREADS_Fan1];
  int NUM_THREADS_Fan2 = gridSize2d * gridSize2d * blockSize2d * blockSize2d;
  pthread_t *threads_Fan2 = new pthread_t[NUM_THREADS_Fan2];
  // begin timing kernels
  struct timeval time_start;
  gettimeofday(&time_start, NULL);
  for (t = 0; t < (Size - 1); t++) {
    // Fan1<<<dimGrid, dimBlock>>>(m_cuda, a_cuda, Size, t);
    // cudaThreadSynchronize();
    {
      // set grid, block dim for F1
      setup_grid_size(grid_size, 1, 1);
      setup_block_size(block_size, 1, 1);
      for (long tid = 0; tid < NUM_THREADS_Fan1; tid++) {
        void *inp = gen_input_Fan1(tid, m, a, Size, t);
        rc = pthread_create(&threads_Fan1[tid], NULL, wrapper_Fan1, inp);
        if (rc) {
          printf("ERROR; return code from pthread_create() is %d\n", rc);
          exit(-1);
        }
      }
      /* Last thing that main() should do */
      for (long tid = 0; tid < NUM_THREADS_Fan1; tid++)
        pthread_join(threads_Fan1[tid], NULL);
    }
    // Fan2<<<dimGridXY, dimBlockXY>>>(m_cuda, a_cuda, b_cuda, Size, Size - t,
    // t); cudaThreadSynchronize();
    {
      // set grid, block dim for F2
      setup_grid_size(gridSize2d, gridSize2d, 1);
      setup_block_size(blockSize2d, blockSize2d, 1);

      for (long tid = 0; tid < NUM_THREADS_Fan2; tid++) {
        void *inp = gen_input_Fan2(tid, m, a, b, Size, Size - t, t);
        rc = pthread_create(&threads_Fan2[t], NULL, wrapper_Fan2, inp);
        if (rc) {
          printf("ERROR; return code from pthread_create() is %d\n", rc);
          exit(-1);
        }
      }
      /* Last thing that main() should do */
      for (long tid = 0; tid < NUM_THREADS_Fan2; tid++)
        pthread_join(threads_Fan2[tid], NULL);
    }
  }
  // end timing kernels
  struct timeval time_end;
  gettimeofday(&time_end, NULL);
  totalKernelTime = (time_end.tv_sec * 1000000 + time_end.tv_usec) -
                    (time_start.tv_sec * 1000000 + time_start.tv_usec);

#ifdef TIMING
  tvsub(&time_end, &tv_kernel_start, &tv);
  kernel_time += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;
#endif
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub() {
  // create a new vector to hold the final answer
  finalVec = (float *)malloc(Size * sizeof(float));
  // solve "bottom up"
  int i, j;
  for (i = 0; i < Size; i++) {
    finalVec[Size - i - 1] = b[Size - i - 1];
    for (j = 0; j < i; j++) {
      finalVec[Size - i - 1] -= *(a + Size * (Size - i - 1) + (Size - j - 1)) *
                                finalVec[Size - j - 1];
    }
    finalVec[Size - i - 1] =
        finalVec[Size - i - 1] / *(a + Size * (Size - i - 1) + (Size - i - 1));
  }
}

void InitMat(float *ary, int nrow, int ncol) {
  int i, j;

  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      fscanf(fp, "%f", ary + Size * i + j);
    }
  }
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol) {
  int i, j;

  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      printf("%8.2f ", *(ary + Size * i + j));
    }
    printf("\n");
  }
  printf("\n");
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(float *ary, int ary_size) {
  int i;

  for (i = 0; i < ary_size; i++) {
    fscanf(fp, "%f", &ary[i]);
  }
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size) {
  int i;
  for (i = 0; i < ary_size; i++) {
    printf("%.2f ", ary[i]);
  }
  printf("\n\n");
}
