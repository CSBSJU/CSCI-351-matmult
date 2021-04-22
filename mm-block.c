/*
Copyright (c) 2016-2020 Jeremy Iverson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#define _XOPEN_SOURCE

/* assert */
#include <assert.h>

#ifdef _OPENMP
/* omp_get_wtime */
#include <omp.h>
#else
#define omp_get_wtime() 0.0
#endif

/* printf, fopen, fclose, fscanf */
#include <stdio.h>

/* EXIT_SUCCESS, malloc, free */
#include <stdlib.h>

/* getopt */
#include <unistd.h>

#define ROWMJR(r,c,nr,nc) (r*nc+c)
#define COLMJR(r,c,nr,nc) (c*nr+r)
/* define access directions for matrices */
#define A(r,c) A[ROWMJR(r,c,n,m)]
#define B(r,c) B[ROWMJR(r,c,m,p)]
#define C(r,c) C[ROWMJR(r,c,n,p)]

#define min(x,y) ((x)<(y)?(x):(y))

extern int optind;

static void
S_matload(char const * const filename, size_t * const np, size_t * const mp,
          double ** const Ap)
{
  size_t n, m;

  FILE * const fp = fopen(filename, "r");
  assert(fp);

  fscanf(fp, "%zu %zu", &n, &m);

  double * const A = malloc(n * m * sizeof(*A));
  assert(A);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      fscanf(fp, "%lf", &A(i, j));
    }
  }

  int const ret = fclose(fp);
  assert(!ret);

  *np = n;
  *mp = m;
  *Ap = A;
}

static void
S_matsave(char const * const filename, size_t const m, size_t const n,
          double const * const A)
{
  FILE * const fp = fopen(filename, "w");
  assert(fp);

  fprintf(fp, "%zu %zu\n", m, n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      fprintf(fp, "%10.4f ", A(i, j));
    }
    fprintf(fp, "\n");
  }

  int const ret = fclose(fp);
  assert(!ret);
}

static void
S_matmult_blk(size_t const n, size_t const m, size_t const p,
              size_t const x, size_t const y, size_t const z,
              double const * const restrict A,
              double const * const restrict B,
              double       * const restrict C)
{
  for (size_t is = 0; is < n; is += y) {   /* block rows of A */
    size_t const ie = min(is + y, n);
    for (size_t js = 0; js < p; js += z) { /* block columns of B */
      size_t const je = min(js + z, p);
      for (size_t ks = 0; ks < m; ks += x) {
        size_t const ke = min(ks + x, m);  /* block dot-product */
        for (size_t i = is; i < ie; ++i) {
          for (size_t j = js; j < je; ++j) {
            double cv = C(i, j);
            for (size_t k = ks; k < ke; ++k) {
              cv += A(i, k) * B(k, j);
            }
            C(i, j) = cv;
          }
        }
      }
    }
  }
}

int
main(int argc, char * argv[])
{
  int opt, num_threads;
  size_t n, m, p, y, x, z;
  double * A, * B, * C;

  x = y = z = 1;
  num_threads = 1;
  while (-1 != (opt=getopt(argc, argv, "x:y:z:t:"))) {
    switch (opt) {
      case 't':
      num_threads = atoi(optarg);
      break;

      case 'x':
      x = (size_t)atol(optarg);
      break;

      case 'y':
      y = (size_t)atol(optarg);
      break;

      case 'z':
      z = (size_t)atol(optarg);
      break;

      default: /* '?' */
      fprintf(stderr, "Usage: %s [-x x dim] [-y y dim] [-z z dim] "\
        "[-t num_threads]\n", argv[0]);
      return EXIT_FAILURE;
    }
  }

  assert(optind == argc - 3);
  char const * const fn_a = argv[optind];
  char const * const fn_b = argv[optind+1];
  char const * const fn_c = argv[optind+2];

  /* Validate input. */
  assert(num_threads > 0);
  assert(x > 0);
  assert(y > 0);
  assert(z > 0);
  assert(fn_a);
  assert(fn_b);
  assert(fn_c);

  S_matload(fn_a, &n, &m, &A);
  S_matload(fn_b, &m, &p, &B);

  C = calloc(n * p, sizeof(*C));
  assert(C);

  /* Fix-up input. */
  y = (y < n) ? y : n;
  x = (x < m) ? x : m;
  z = (z < p) ? z : p;

  double const ts = omp_get_wtime();
  S_matmult_blk(n, m, p, y, x, z, A, B, C);
  double const te = omp_get_wtime();
  double const t1 = te - ts;

  printf("Matrix operation time: %0.04fs\n", t1);

  S_matsave(fn_c, n, p, C);

  /* Free remaining memory. */
  free(A);
  free(B);
  free(C);

  return EXIT_SUCCESS;
}
