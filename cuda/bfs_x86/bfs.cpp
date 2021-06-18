/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology -
 Hyderabad. All rights reserved.

  Permission to use, copy, modify and distribute this software and its
 documentation for educational purpose is hereby granted without fee, provided
 that the above copyright notice and this permission notice appear in all copies
 of this software and that you do not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS,
 IMPLIED OR OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include "library.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TIMING
#include "timing.h"
#endif

#define MAX_THREADS_PER_BLOCK 512

int no_of_nodes;
int edge_list_size;
FILE *fp;

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

// Structure to hold a node information
struct Node {
  int starting;
  int no_of_edges;
};

extern "C" {
void *_Z6KernelP4NodePiPbS2_S2_S1_i_wrapper(void *);
void *_Z7Kernel2PbS_S_S_i_wrapper(void *);
}

void *wrapper_kernel(void *p) {
  int **ret = (int **)p;
  int tid = *(ret[0]);
  setup_idx(tid);
  _Z6KernelP4NodePiPbS2_S2_S1_i_wrapper((void *)(ret + 1));
  return NULL;
}

void *wrapper_kernel2(void *p) {
  int **ret = (int **)p;
  int tid = *(ret[0]);
  setup_idx(tid);
  _Z7Kernel2PbS_S_S_i_wrapper((void *)(ret + 1));
  return NULL;
}

void *gen_input_kernel(int tid, Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes) {
  int **ret = new int *[8];

  int *p0 = new int;
  *p0 = tid;
  ret[0] = (int *)p0;

  Node **p1 = new Node *;
  *p1 = g_graph_nodes;
  ret[1] = (int *)(p1);

  int **p2 = new int *;
  *p2 = g_graph_edges;
  ret[2] = (int *)(p2);

  bool **p3 = new bool *;
  *p3 = g_graph_mask;
  ret[3] = (int *)(p3);

  bool **p4 = new bool *;
  *p4 = g_updating_graph_mask;
  ret[4] = (int *)(p4);

  bool **p5 = new bool *;
  *p5 = g_graph_visited;
  ret[5] = (int *)(p5);

  int **p6 = new int *;
  *p6 = g_cost;
  ret[6] = (int *)(p6);

  int *p7 = new int;
  *p7 = no_of_nodes;
  ret[7] = (int *)p7;

  return (void *)ret;
}

void *gen_input_kernel2(int tid, bool *g_graph_mask,
                        bool *g_updating_graph_mask, bool *g_graph_visited,
                        bool *g_over, int no_of_nodes) {
  int **ret = new int *[6];

  int *p0 = new int;
  *p0 = tid;
  ret[0] = (int *)p0;

  bool **p1 = new bool *;
  *p1 = g_graph_mask;
  ret[1] = (int *)(p1);

  bool **p2 = new bool *;
  *p2 = g_updating_graph_mask;
  ret[2] = (int *)(p2);

  bool **p3 = new bool *;
  *p3 = g_graph_visited;
  ret[3] = (int *)(p3);

  bool **p4 = new bool *;
  *p4 = g_over;
  ret[4] = (int *)(p4);

  int *p5 = new int;
  *p5 = no_of_nodes;
  ret[5] = (int *)p5;

  return (void *)ret;
}

void BFSGraph(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  no_of_nodes = 0;
  edge_list_size = 0;
  BFSGraph(argc, argv);
}

void Usage(int argc, char **argv) {

  fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
}
////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char **argv) {

  char *input_f;
  if (argc != 2) {
    Usage(argc, argv);
    exit(0);
  }

  input_f = argv[1];
  printf("Reading File\n");
  // Read in Graph from a file
  fp = fopen(input_f, "r");
  if (!fp) {
    printf("Error Reading graph file\n");
    return;
  }

  int source = 0;

  fscanf(fp, "%d", &no_of_nodes);

  int num_of_blocks = 1;
  int num_of_threads_per_block = no_of_nodes;

  // Make execution Parameters according to the number of nodes
  // Distribute threads across multiple Blocks if necessary
  if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
    num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
    num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
  }

  // allocate host memory
  Node *h_graph_nodes = (Node *)malloc(sizeof(Node) * no_of_nodes);
  bool *h_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
  bool *h_updating_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
  bool *h_graph_visited = (bool *)malloc(sizeof(bool) * no_of_nodes);

  int start, edgeno;
  // initalize the memory
  for (unsigned int i = 0; i < no_of_nodes; i++) {
    fscanf(fp, "%d %d", &start, &edgeno);
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].no_of_edges = edgeno;
    h_graph_mask[i] = false;
    h_updating_graph_mask[i] = false;
    h_graph_visited[i] = false;
  }

  // read the source node from the file
  fscanf(fp, "%d", &source);
  source = 0;

  // set the source node as true in the mask
  h_graph_mask[source] = true;
  h_graph_visited[source] = true;

  fscanf(fp, "%d", &edge_list_size);

  int id, cost;
  int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
  for (int i = 0; i < edge_list_size; i++) {
    fscanf(fp, "%d", &id);
    fscanf(fp, "%d", &cost);
    h_graph_edges[i] = id;
  }

  if (fp)
    fclose(fp);

  printf("Read File\n");

#ifdef TIMING
  gettimeofday(&tv_total_start, NULL);
#endif
  // Copy the Node list to device memory
  Node *d_graph_nodes = h_graph_nodes;

  // Copy the Edge List to device Memory
  int *d_graph_edges = h_graph_edges;

  // Copy the Mask to device memory
  bool *d_graph_mask = h_graph_mask;

  bool *d_updating_graph_mask = h_updating_graph_mask;

  // Copy the Visited nodes array to device memory
  bool *d_graph_visited = h_graph_visited;

  // allocate mem for the result on host side
  int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
  for (int i = 0; i < no_of_nodes; i++)
    h_cost[i] = -1;
  h_cost[source] = 0;

  // allocate device memory for result
  int *d_cost = h_cost;

  // make a bool to check if the execution is over
  bool stop;
  bool *d_over = &stop;

#ifdef TIMING
  gettimeofday(&tv_mem_alloc_end, NULL);
  tvsub(&tv_mem_alloc_end, &tv_total_start, &tv);
  h2d_time = tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;
#endif

  printf("Copied Everything to GPU memory\n");

  // setup execution parameters
  printf("block %d thread %d\n", num_of_blocks, num_of_threads_per_block);
  setup_grid_size(num_of_blocks, 1, 1);
  setup_block_size(num_of_threads_per_block, 1, 1);
  int NUM_THREADS = num_of_blocks * num_of_threads_per_block;
  pthread_t *threads = new pthread_t[NUM_THREADS];
  int rc;
  int *thread_id = new int[NUM_THREADS];
  //   dim3 grid(num_of_blocks, 1, 1);
  //   dim3 threads(num_of_threads_per_block, 1, 1);

  int k = 0;
  printf("Start traversing the tree\n");

  // Call the Kernel untill all the elements of Frontier are not false
  do {
    // if no thread changes this value then the loop stops
    stop = false;
#ifdef TIMING
    gettimeofday(&tv_h2d_start, NULL);
#endif

#ifdef TIMING
    gettimeofday(&tv_h2d_end, NULL);
    tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
    h2d_time += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;
#endif

    for (long t = 0; t < NUM_THREADS; t++) {
      void *inp = gen_input_kernel(t, d_graph_nodes, d_graph_edges,
                                   d_graph_mask, d_updating_graph_mask,
                                   d_graph_visited, d_cost, no_of_nodes);
      rc = pthread_create(&threads[t], NULL, wrapper_kernel, inp);
      if (rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
      }
    }

    /* Last thing that main() should do */
    for (long t = 0; t < NUM_THREADS; t++)
      pthread_join(threads[t], NULL);
    // Kernel<<<grid, threads, 0>>>(d_graph_nodes, d_graph_edges, d_graph_mask,
    //                              d_updating_graph_mask, d_graph_visited,
    //                              d_cost, no_of_nodes);
    // check if kernel execution generated and error
    for (long t = 0; t < NUM_THREADS; t++) {
      void *inp = gen_input_kernel2(t, d_graph_mask, d_updating_graph_mask,
                                    d_graph_visited, d_over, no_of_nodes);
      rc = pthread_create(&threads[t], NULL, wrapper_kernel2, inp);
      if (rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
      }
    }

    /* Last thing that main() should do */
    for (long t = 0; t < NUM_THREADS; t++)
      pthread_join(threads[t], NULL);
      // Kernel2<<<grid, threads, 0>>>(d_graph_mask, d_updating_graph_mask,
      //                               d_graph_visited, d_over, no_of_nodes);
      // check if kernel execution generated and error

#ifdef TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_h2d_end, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;
#endif

#ifdef TIMING
    gettimeofday(&tv_d2h_end, NULL);
    tvsub(&tv_d2h_end, &tv_kernel_end, &tv);
    d2h_time += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;
#endif

    k++;
  } while (stop);

  printf("Kernel Executed %d times\n", k);

  // copy result from device to host
#ifdef TIMING
  gettimeofday(&tv_d2h_start, NULL);
#endif

#ifdef TIMING
  gettimeofday(&tv_d2h_end, NULL);
  tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
  d2h_time += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;
#endif

  // Store the result into a file
  FILE *fpo = fopen("result.txt", "w");
  for (int i = 0; i < no_of_nodes; i++)
    fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
  fclose(fpo);
  printf("Result stored in result.txt\n");

  // cleanup memory
  free(h_graph_nodes);
  free(h_graph_edges);
  free(h_graph_mask);
  free(h_updating_graph_mask);
  free(h_graph_visited);
  free(h_cost);
#ifdef TIMING
  gettimeofday(&tv_close_start, NULL);
#endif

#ifdef TIMING
  gettimeofday(&tv_close_end, NULL);
  tvsub(&tv_close_end, &tv_close_start, &tv);
  close_time = tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;
  tvsub(&tv_close_end, &tv_total_start, &tv);
  total_time = tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;

  printf("Init: %f\n", init_time);
  printf("MemAlloc: %f\n", mem_alloc_time);
  printf("HtoD: %f\n", h2d_time);
  printf("Exec: %f\n", kernel_time);
  printf("DtoH: %f\n", d2h_time);
  printf("Close: %f\n", close_time);
  printf("Total: %f\n", total_time);
#endif
}
