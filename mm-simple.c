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

#define ROWMJR(r,c,nr,nc) (r*nc+c)
#define COLMJR(r,c,nr,nc) (c*nr+r)
/* define access directions for matrices */
#define A(r,c) A[ROWMJR(r,c,n,m)]
#define B(r,c) B[ROWMJR(r,c,m,p)]
#define C(r,c) C[ROWMJR(r,c,n,p)]

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
S_matmult(size_t const n, size_t const m, size_t const p,
          double const * const restrict A,
          double const * const restrict B,
          double       * const restrict C)
{
  for (size_t i = 0; i < n; ++i) {      /* rows of A */
    for (size_t j = 0; j < m; ++j) {    /* columns of B */
      double cv = 0.0;
      for (size_t k = 0; k < p; ++k) {  /* dot-product */
        cv += A(i, k) * B(k, j);
      }
      C(i, j) = cv;
    }
  }
}

int
main(int argc, char * argv[])
{
  size_t n, m, p;
  double * A, * B, * C;

  /* Validate command line arguments. */
  assert(4 == argc);

  /* ... */
  char const * const fn_a = argv[1];
  char const * const fn_b = argv[2];
  char const * const fn_c = argv[3];

  /* Validate input. */
  assert(fn_a);
  assert(fn_b);
  assert(fn_c);

  /* Load matrices. */
  S_matload(fn_a, &n, &m, &A);
  S_matload(fn_b, &m, &p, &B);

  /* Allocate memory for output matrix. */
  C = calloc(n * p, sizeof(*C));
  assert(C);

  double const ts = omp_get_wtime();
  S_matmult(n, m, p, A, B, C);
  double const te = omp_get_wtime();
  double const t1 = te - ts;

  printf("Matrix operation time: %0.04fs\n", t1);

  /* Write solution. */
  S_matsave(fn_c, n, p, C);

  /* Free remaining memory. */
  free(A);
  free(B);
  free(C);

  return EXIT_SUCCESS;
}
