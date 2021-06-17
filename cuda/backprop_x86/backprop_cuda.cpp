

// includes, system
#include "library.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// includes, kernels
#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern "C" void bpnn_layerforward(float *l1, float *l2, float **conn, int n1,
                                  int n2);

extern "C" void bpnn_output_error(float *delta, float *target, float *output,
                                  int nj, float *err);

extern "C" void bpnn_hidden_error(float *delta_h, int nh, float *delta_o,
                                  int no, float **who, float *hidden,
                                  float *err);

extern "C" void bpnn_adjust_weights(float *delta, int ndelta, float *ly,
                                    int nly, float **w, float **oldw);

extern "C" int setup(int argc, char **argv);

extern "C" float **alloc_2d_dbl(int m, int n);

extern "C" float squash(float x);

extern "C" {
void *_Z22bpnn_layerforward_CUDAPfS_S_S_ii_wrapper(void *);
void *_Z24bpnn_adjust_weights_cudaPfiS_iS_S__wrapper(void *);
}

void *wrapper_func_forward(void *p) {
  int **ret = (int **)p;
  int tid = *(ret[0]);
  setup_idx(tid);
  _Z22bpnn_layerforward_CUDAPfS_S_S_ii_wrapper((void *)(ret + 1));
  return NULL;
}

void *wrapper_func_adjust(void *p) {
  int **ret = (int **)p;
  int tid = *(ret[0]);
  setup_idx(tid);
  _Z24bpnn_adjust_weights_cudaPfiS_iS_S__wrapper((void *)(ret + 1));
  return NULL;
}

void *gen_input_adjust(int tid, float *delta, int hid, float *ly, int in,
                       float *w, float *oldw) {
  int **ret = new int *[7];

  int *p0 = new int;
  *p0 = tid;
  ret[0] = (int *)p0;
  float **p1 = new float *;
  *p1 = delta;
  ret[1] = (int *)(p1);
  int *p2 = new int;
  *p2 = hid;
  ret[2] = (int *)p2;
  float **p3 = new float *;
  *p3 = ly;
  ret[3] = (int *)(p3);
  int *p4 = new int;
  *p4 = in;
  ret[4] = (int *)p4;
  float **p5 = new float *;
  *p5 = w;
  ret[5] = (int *)(p5);
  float **p6 = new float *;
  *p6 = oldw;
  ret[6] = (int *)(p6);

  return (void *)ret;
}

void *gen_input_forward(int tid, float *input_cuda, float *output_hidden_cuda,
                        float *input_hidden_cuda, float *hidden_partial_sum,
                        int in, int hid) {
  int **ret = new int *[7];

  int *p0 = new int;
  *p0 = tid;
  ret[0] = (int *)p0;

  float **p1 = new float *;
  *p1 = input_cuda;
  ret[1] = (int *)(p1);

  float **p2 = new float *;
  *p2 = output_hidden_cuda;
  ret[2] = (int *)(p2);

  float **p3 = new float *;
  *p3 = input_hidden_cuda;
  ret[3] = (int *)(p3);

  float **p4 = new float *;
  *p4 = hidden_partial_sum;
  ret[4] = (int *)(p4);

  int *p5 = new int;
  *p5 = in;
  ret[5] = (int *)p5;

  int *p6 = new int;
  *p6 = hid;
  ret[6] = (int *)p6;
  return (void *)ret;
}

double gettime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { setup(argc, argv); }

extern "C" void bpnn_train_cuda(BPNN *net, float *eo, float *eh) {
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

#ifdef GPU
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;

  input_weights_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  input_weights_prev_one_dim =
      (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));

  // this preprocessing stage is added to correct the bugs of wrong memcopy
  // using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
      m++;
    }
  }

  input_cuda = (float *)malloc((in + 1) * sizeof(float));
  output_hidden_cuda = (float *)malloc((hid + 1) * sizeof(float));
  input_hidden_cuda = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  hidden_partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));
  // cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
  // cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  // cudaMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) *
  // sizeof(float)); cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH
  // * sizeof(float));

#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in,
                    hid);

#endif

#ifdef GPU

  printf("Performing GPU computation\n");
  printf("block:%d\n", num_blocks);

  // printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);

  memcpy(input_cuda, net->input_units, (in + 1) * sizeof(float));
  memcpy(input_hidden_cuda, input_weights_one_dim,
         (in + 1) * (hid + 1) * sizeof(float));

  // cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
  //            (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  setup_grid_size(1, num_blocks, 1);
  setup_block_size(16, 16, 1);

  int NUM_THREADS = 1 * num_blocks * 16 * 16;
  pthread_t *threads = new pthread_t[NUM_THREADS];
  int rc;
  int *thread_id = new int[NUM_THREADS];
  printf("before\n");
  for (long t = 0; t < NUM_THREADS; t++) {
    void *inp =
        gen_input_forward(t, input_cuda, output_hidden_cuda, input_hidden_cuda,
                          hidden_partial_sum, in, hid);
    rc = pthread_create(&threads[t], NULL, wrapper_func_forward, inp);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Last thing that main() should do */
  for (long t = 0; t < NUM_THREADS; t++)
    pthread_join(threads[t], NULL);
  // dim3 grid(1, num_blocks);
  // dim3 threads(16, 16);
  // bpnn_layerforward_CUDA<<<grid, threads>>>(input_cuda, output_hidden_cuda,
  //                                           input_hidden_cuda,
  //                                           hidden_partial_sum, in, hid);

  // cudaThreadSynchronize();

  // cudaError_t error = cudaGetLastError();
  // if (error != cudaSuccess) {
  //   printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
  //   exit(EXIT_FAILURE);
  // }
  memcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  // cudaMemcpy(partial_sum, hidden_partial_sum,
  //            num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j - 1];
    }
    sum += net->input_weights[0][j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
#endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                    hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                    &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                    net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                      net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                      net->input_weights, net->input_prev_weights);

#endif

#ifdef GPU

  hidden_delta_cuda = (float *)malloc((hid + 1) * sizeof(float));
  input_prev_weights_cuda =
      (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  // cudaMalloc((void **)&hidden_delta_cuda, (hid + 1) * sizeof(float));
  // cudaMalloc((void **)&input_prev_weights_cuda,
  //            (in + 1) * (hid + 1) * sizeof(float));
  memcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float));
  memcpy(input_prev_weights_cuda, input_weights_prev_one_dim,
         (in + 1) * (hid + 1) * sizeof(float));
  memcpy(input_hidden_cuda, input_weights_one_dim,
         (in + 1) * (hid + 1) * sizeof(float));

  // cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim,
  //            (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
  //            (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  for (long t = 0; t < NUM_THREADS; t++) {
    void *inp = gen_input_adjust(t, hidden_delta_cuda, hid, input_cuda, in,
                                 input_hidden_cuda, input_prev_weights_cuda);
    rc = pthread_create(&threads[t], NULL, wrapper_func_adjust, inp);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }
  /* Last thing that main() should do */
  for (long t = 0; t < NUM_THREADS; t++)
    pthread_join(threads[t], NULL);
  // bpnn_adjust_weights_cuda<<<grid, threads>>>(hidden_delta_cuda, hid,
  //                                             input_cuda, in,
  //                                             input_hidden_cuda,
  //                                             input_prev_weights_cuda);
  memcpy(net->input_units, input_cuda, (in + 1) * sizeof(float));
  memcpy(input_weights_one_dim, input_hidden_cuda,
         (in + 1) * (hid + 1) * sizeof(float));
  // cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float),
  //            cudaMemcpyDeviceToHost);
  // cudaMemcpy(input_weights_one_dim, input_hidden_cuda,
  //            (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);

  // cudaFree(input_cuda);
  // cudaFree(output_hidden_cuda);
  // cudaFree(input_hidden_cuda);
  // cudaFree(hidden_partial_sum);
  // cudaFree(input_prev_weights_cuda);
  // cudaFree(hidden_delta_cuda);

  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);

#endif
}
