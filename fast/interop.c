#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef float f32;
typedef double f64;

typedef int32_t i32;
typedef int64_t i64;

typedef int8_t i8;

#define MEMREF_STRUCT_DEF(dtype, rank) struct dtype ## _memref_r_ ## rank {    \
  dtype *allocated;                          \
  dtype *aligned;                            \
  intptr_t offset;                           \
  intptr_t sizes[rank];                      \
  intptr_t strides[rank];                    \
};                                           \

#ifndef OUTFILE_NAME
#define OUTFILE_NAME "result.data"
#endif

#ifndef INFILE_NAME
#define INFILE_NAME "input.data"
#endif

// define memref rank 1 to 3 for f32, f64, i32, i64
// these will be named f32_memref_r_2 for example

MEMREF_STRUCT_DEF(i8, 1)
MEMREF_STRUCT_DEF(i8, 2)
MEMREF_STRUCT_DEF(i8, 3)

MEMREF_STRUCT_DEF(f32, 1)
MEMREF_STRUCT_DEF(f32, 2)
MEMREF_STRUCT_DEF(f32, 3)

MEMREF_STRUCT_DEF(i32, 1)
MEMREF_STRUCT_DEF(i32, 2)
MEMREF_STRUCT_DEF(i32, 3)

MEMREF_STRUCT_DEF(f64, 1)
MEMREF_STRUCT_DEF(f64, 2)
MEMREF_STRUCT_DEF(f64, 3)

MEMREF_STRUCT_DEF(i64, 1)
MEMREF_STRUCT_DEF(i64, 2)
MEMREF_STRUCT_DEF(i64, 3)


// code for packing/unpacking memrefs to/from args
// please don't look at this too closely here:
#define REP0(X)
#define REP1(X) X ## _1
#define REP2(X) REP1(X) , X ## _2
#define REP3(X) REP2(X) , X ## _3
#define REP4(X) REP3(X) , X ## _4
#define REP5(X) REP4(X) , X ## _5
#define REP6(X) REP5(X) , X ## _6
#define REP7(X) REP6(X) , X ## _7
#define REP8(X) REP7(X) , X ## _8
#define REP9(X) REP8(X) , X ## _9
#define REP10(X) REP9(X) , X ## _10


#define UNPACK_REP0(X)
#define UNPACK_REP1(X) X[0]
#define UNPACK_REP2(X) UNPACK_REP1(X) , X[1]
#define UNPACK_REP3(X) UNPACK_REP2(X) , X[2]
#define UNPACK_REP4(X) UNPACK_REP3(X) , X[3]
#define UNPACK_REP5(X) UNPACK_REP4(X) , X[4]
#define UNPACK_REP6(X) UNPACK_REP5(X) , X[5]
#define UNPACK_REP7(X) UNPACK_REP6(X) , X[6]
#define UNPACK_REP8(X) UNPACK_REP7(X) , X[7]
#define UNPACK_REP9(X) UNPACK_REP8(X) , X[8]
#define UNPACK_REP10(X) UNPACK_REP9(X) , X[9]

#define UNPACK_NO_COMMA_REP0(X)
#define UNPACK_NO_COMMA_REP1(X) X[0]
#define UNPACK_NO_COMMA_REP2(X) UNPACK_NO_COMMA_REP1(X) X[1]
#define UNPACK_NO_COMMA_REP3(X) UNPACK_NO_COMMA_REP2(X) X[2]
#define UNPACK_NO_COMMA_REP4(X) UNPACK_NO_COMMA_REP3(X) X[3]
#define UNPACK_NO_COMMA_REP5(X) UNPACK_NO_COMMA_REP4(X) X[4]
#define UNPACK_NO_COMMA_REP6(X) UNPACK_NO_COMMA_REP5(X) X[5]
#define UNPACK_NO_COMMA_REP7(X) UNPACK_NO_COMMA_REP6(X) X[6]
#define UNPACK_NO_COMMA_REP8(X) UNPACK_NO_COMMA_REP7(X) X[7]
#define UNPACK_NO_COMMA_REP9(X) UNPACK_NO_COMMA_REP8(X) X[8]
#define UNPACK_NO_COMMA_REP10(X) UNPACK_NO_COMMA_REP9(X) X[9]

// oh god, this is unholy:

#define MEMREF_AS_ARGS_DEF(prefix, dtype, rank) dtype * prefix ## allocated, dtype * prefix ## aligned, intptr_t prefix ## offset, REP ## rank (intptr_t prefix ## sizes), REP ## rank (intptr_t prefix ## strides) 

#define COLLECT_MEMREF_ARGS_INTO(prefix, dtype, rank, name) struct dtype ## _memref_r_ ## rank name = { prefix ## allocated, prefix ## aligned, prefix ## offset, REP ## rank (prefix ## sizes), REP ## rank (prefix ## strides) }

#define MEMREF_TO_ARGS(ref, rank) ref.allocated, ref.aligned, ref.offset, UNPACK_REP ## rank (ref.sizes), UNPACK_REP ## rank (ref.strides)

// dumping memref macros:

#if NODUMP
#define DUMP_MEMREF(fname, name, dtype, rank) \
  {                                           \
    printf("Skipping output dumping!\n");     \
  }                                           
#else 
#define DUMP_MEMREF(fname, name, dtype, rank)                                         \
  {                                                                                   \
    FILE *f = fopen(fname, "w");                                                      \
    fwrite(name.aligned, sizeof(dtype), 1 UNPACK_NO_COMMA_REP##rank(*name.sizes), f); \
    fclose(f);                                                                        \
  }
#endif

// linearized accesses:

#define LIN_ACCESS2(ref, x, y) ref.aligned[(x) * ref.sizes[1] + (y)]
#define LIN_ACCESS3(ref, x, y, z) ref.aligned[(x) * ref.sizes[1] * ref.sizes[2] + (y) * ref.sizes[2] + (z)]

// dumping methods:

#define GENERATE_DUMPING_FUNC(dtype, rank) void dump_memref_ ## dtype ## _rank_ ## rank (MEMREF_AS_ARGS_DEF(my, dtype, rank)) { \
    COLLECT_MEMREF_ARGS_INTO(my, dtype, rank, my_memref); \
    DUMP_MEMREF(OUTFILE_NAME, my_memref, dtype, rank) \
}

// generate function defs:

GENERATE_DUMPING_FUNC(f32, 1)
GENERATE_DUMPING_FUNC(f32, 2)
GENERATE_DUMPING_FUNC(f32, 3)

GENERATE_DUMPING_FUNC(f64, 1)
GENERATE_DUMPING_FUNC(f64, 2)
GENERATE_DUMPING_FUNC(f64, 3)

GENERATE_DUMPING_FUNC(i32, 1)
GENERATE_DUMPING_FUNC(i32, 2)
GENERATE_DUMPING_FUNC(i32, 3)

GENERATE_DUMPING_FUNC(i64, 1)
GENERATE_DUMPING_FUNC(i64, 2)
GENERATE_DUMPING_FUNC(i64, 3)

/*
This file provides the following functions for MLIR:

func.func private @dump_memref_i32_rank_1(memref<?xi32>) -> ()
func.func private @dump_memref_f32_rank_1(memref<?xf32>) -> ()
func.func private @dump_memref_i64_rank_1(memref<?xi64>) -> ()
func.func private @dump_memref_f64_rank_1(memref<?xf64>) -> ()

func.func private @dump_memref_i32_rank_2(memref<?x?xi32>) -> ()
func.func private @dump_memref_f32_rank_2(memref<?x?xf32>) -> ()
func.func private @dump_memref_i64_rank_2(memref<?x?xi64>) -> ()
func.func private @dump_memref_f64_rank_2(memref<?x?xf64>) -> ()

func.func private @dump_memref_i32_rank_3(memref<?x?x?xi32>) -> () 
func.func private @dump_memref_f32_rank_3(memref<?x?x?xf32>) -> () 
func.func private @dump_memref_i64_rank_3(memref<?x?x?xi64>) -> () 
func.func private @dump_memref_f64_rank_3(memref<?x?x?xf64>) -> () 

You can call them using:

func.call @dump_memref_f64_rank_3(%ref) : (memref<?x?x?xi64>) -> ()

or any other signature as provided above

The output file will be outfile

*/

const struct i8_memref_r_1 load_memref(char* fname, size_t length) {
  void* ptr = aligned_alloc(64, length);
  struct i8_memref_r_1 ref = {ptr, ptr, 1, length, 1};
  FILE* f = fopen(fname,"r");
  size_t num = fread(ptr, 1, length, f);
  if (num != length) {
    printf("WARN: file read failed! Only read %ld bytes!\n", num);
  }
  fclose(f);
  return ref;
}

struct i8_memref_r_1 load_input(size_t length) {
  return load_memref(INFILE_NAME, length);
}

void print_i32(int n)
{
  printf("%d\n", n);
}

extern int MPI_Comm_rank(int comm, int *rank);

void print_halo_send_info(int dest, int ex, i64 x0, i64 y0, i64 h, i64 w) {
  int rank;
  MPI_Comm_rank(1140850688, &rank);
  i64 x1 = x0 + h;
  i64 y1 = y0 + w;
  printf("MPI send ex%i [%li:%li,%li:%li] %i -> %i\n",ex, x0, y0, x1, y1,  rank, dest);
}
void print_halo_recv_info(int src, int ex, i64 x0, i64 y0, i64 h, i64 w) {
  int rank;
  MPI_Comm_rank(1140850688, &rank);
  i64 x1 = x0 + h;
  i64 y1 = y0 + w;
  printf("MPI recv ex%i [%li:%li,%li:%li] %i <- %i\n",ex, x0, y0, x1, y1, rank, src);
}

i64 timer_start() {
  // return epoch in ms
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  //return (t.tv_sec * 1e3) + ((i64) t.tv_nsec / 1e6);
  i64 msecs = (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
  printf("Timestamp is: %.2f \n", msecs * 1e-3);
  return msecs;
}

void timer_end(i64 start) {
  // print time elapsed time
  i64 end = timer_start();
  i64 elapsed_time = (end - start);
  printf("Elapsed time is: %.17f secs\n", elapsed_time * 1e-3);
}


