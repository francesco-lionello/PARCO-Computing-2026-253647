/*
spmv.c — Read Matrix Market (.mtx) -> COO -> CSR (sequential)

Compile:
  gcc -O3 -march=native -Wall -Wextra -fopenmp -o spmv spmv.c -lrt

Note:
  - Support Matrix Market: "matrix coordinate real {general|symmetric}"
  - CSR: row-parallel with OpenMP (no race) + hint SIMD 

*/

#include <stdio.h>   // printf, FILE
#include <stdlib.h>  // malloc, free, atoi, strtoull, qsort, rand
#include <stdint.h>  // size_t
#include <string.h>  // memset, memcpy, strncmp, sscanf
#include <math.h>    // fabsf, NAN
#include <time.h>    // clock_gettime, time
#include <errno.h>   // errno, strerror

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------- TIMING ---------------- */
static inline double now_sec(void){
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ---------------- UTILS ---------------- */

static float max_abs_err(const float *x, const float *y, size_t n){
    float e = 0.0f;
    for (size_t i=0;i<n;++i){
        float d = fabsf(x[i]-y[i]);
        if (d > e) e = d;
    }
    return e;
}

static void fill_random(float *a, size_t n){
    for(size_t i=0;i<n;++i) a[i] = (float)rand()/RAND_MAX - 0.5f;
}

static int cmp_double(const void *a, const void *b){
    double da = *(const double*)a;
    double db = *(const double*)b;

    if(da < db) return -1;
    else if(da > db) return 1;
    else return 0;
}

static double percentile(double *vals, size_t n, double p){
    if (n == 0) return NAN;
    if (n == 1) return vals[0];

    // Ordina IN-PLACE
    qsort(vals, n, sizeof(*vals), cmp_double);

    // Clamping p 
    if (p < 0.0)   p = 0.0;
    if (p > 100.0) p = 100.0;

    // Rank
    const double r = (p/100.0) * (double)(n-1);
    const size_t lo = (size_t)floor(r);
    const size_t hi = (size_t)ceil(r);

    if (lo == hi) return vals[lo];

    const double w = r - (double)lo;
    return (1.0 - w) * vals[lo] + w * vals[hi];
}


/* ---------------- COO ---------------- */
typedef struct {
    size_t M, N, nnz;
    size_t *I, *J;  // 0-based
    float  *V;      // values
} coo_t;

static void coo_free(coo_t *A){
    free(A->I); free(A->J); free(A->V);
    memset(A, 0, sizeof(*A));
}

// COO3 for sorting COO by (i,j)
typedef struct { size_t i,j; float v; } coo3_t;

static int cmp_coo3(const void* a, const void* b){
    const coo3_t *x = (const coo3_t*)a;
    const coo3_t *y = (const coo3_t*)b;
    if (x->i < y->i) return -1;
    if (x->i > y->i) return 1;
    if (x->j < y->j) return -1;
    if (x->j > y->j) return 1;
    return 0;
}

/* Matrix Market reader: "matrix coordinate real {general|symmetric}" */
static int mm_read_coo(const char *path, coo_t *A){
    memset(A,0,sizeof(*A));

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"open %s failed: %s\n", path, strerror(errno)); return -1; }

    char line[1024];
    if(!fgets(line,sizeof(line),f)){ fprintf(stderr,"file vuoto\n"); fclose(f); return -1; }

    if (strncmp(line,"%%MatrixMarket",14)!=0 || !strstr(line,"matrix") ||
        !strstr(line,"coordinate") || !strstr(line,"real")) {
        fprintf(stderr,"Support: matrix coordinate real {general|symmetric}\n");
        fclose(f); return -1;
    }

    int is_sym = strstr(line,"symmetric") != NULL;

    // skip comments
    do{
        if(!fgets(line,sizeof(line),f)){ fprintf(stderr,"miss row dimension\n"); fclose(f); return -1; }
    }while(line[0]=='%' || line[0]=='\n' || line[0]=='\r');

    size_t m=0,n=0,k=0;
    if (sscanf(line,"%zu %zu %zu",&m,&n,&k)!=3){ fprintf(stderr,"parse size\n"); fclose(f); return -1; }

    // buffer temporary
    size_t *I0=NULL,*J0=NULL; double *V0=NULL;
    if (posix_memalign((void**)&I0,64,k*sizeof(size_t)) ||
        posix_memalign((void**)&J0,64,k*sizeof(size_t)) ||
        posix_memalign((void**)&V0,64,k*sizeof(double))) {
        fprintf(stderr,"posix_memalign tmp\n"); fclose(f); return -1;
    }

    for(size_t t=0; t<k; ++t){
        size_t ii=0, jj=0; 
        double vv=0.0;

        if (fscanf(f,"%zu %zu %lf",&ii,&jj,&vv) != 3){
            fprintf(stderr,"parse entry %zu\n", t);
            free(I0); free(J0); free(V0); fclose(f); return -1;
        }
        if (ii==0 || jj==0 || ii>m || jj>n){
            fprintf(stderr,"indice fuori range (entry %zu)\n", t);
            free(I0); free(J0); free(V0); fclose(f); return -1;
        }
        I0[t]=ii-1; 
        J0[t]=jj-1; 
        V0[t]=vv;
    }
    fclose(f);

    // expansion if symmetric
    size_t dup=0;

    if (is_sym){ 
        for(size_t t=0;t<k;++t) 
            if(I0[t]!=J0[t]) dup++;
    }
    size_t k2 = k + dup;

    if (posix_memalign((void**)&A->I,64,k2*sizeof(size_t)) ||
        posix_memalign((void**)&A->J,64,k2*sizeof(size_t)) ||
        posix_memalign((void**)&A->V,64,k2*sizeof(float))) {
        fprintf(stderr,"posix_memalign COO\n");
        free(I0); free(J0); free(V0); return -1;
    }
    A->M=m; A->N=n; A->nnz=k2;

    size_t w=0;
    for(size_t t=0;t<k;++t){
        A->I[w]=I0[t]; A->J[w]=J0[t]; A->V[w]=(float)V0[t]; w++;
        if (is_sym && I0[t]!=J0[t]){
            A->I[w]=J0[t]; A->J[w]=I0[t]; A->V[w]=(float)V0[t]; w++;
        }
    }

    free(I0); free(J0); free(V0);
    return 0;
}

static void coo_sort_and_merge(coo_t *A){
    if (A->nnz==0) return;

    coo3_t *T = (coo3_t*)malloc(A->nnz*sizeof(*T));
    for (size_t k=0;k<A->nnz;++k){ T[k].i=A->I[k]; T[k].j=A->J[k]; T[k].v=A->V[k]; }
    qsort(T, A->nnz, sizeof(*T), cmp_coo3);
    size_t w=0;

    for (size_t k=0;k<A->nnz;){
        size_t i=T[k].i, j=T[k].j; float s=T[k].v; ++k;
        while (k<A->nnz && T[k].i==i && T[k].j==j){ s += T[k].v; ++k; }
        T[w].i=i; T[w].j=j; T[w].v=s; ++w;
    }
    for (size_t k=0;k<w;++k){ A->I[k]=T[k].i; A->J[k]=T[k].j; A->V[k]=T[k].v; }
    A->nnz = w; 
    free(T);
}

/* ---------------- CSR ---------------- */
typedef struct {
    size_t M, N, nnz;
    size_t *row_ptr; // M+1
    size_t *col_idx; // nnz
    float  *val;     // nnz
} csr_t;

// free CSR
static void csr_free(csr_t *A){
    free(A->row_ptr); free(A->col_idx); free(A->val);
    memset(A,0,sizeof(*A));
}

/* COO -> CSR (no sort/merge) */
static int coo_to_csr(const coo_t *C, csr_t *A){
    memset(A,0,sizeof(*A));
    A->M=C->M; A->N=C->N; A->nnz=C->nnz;

    const size_t rows=A->M, nnz=A->nnz;
    const size_t bytes_row=(rows+1)*sizeof(size_t);

    if (posix_memalign((void**)&A->row_ptr,64,bytes_row) ||
        posix_memalign((void**)&A->col_idx,64,nnz*sizeof(size_t)) ||
        posix_memalign((void**)&A->val,    64,nnz*sizeof(float))) {
        fprintf(stderr,"posix_memalign CSR failed\n"); return -1;
    }
    memset(A->row_ptr,0,bytes_row);

    // 1) count for row
    for(size_t k=0; k<nnz; ++k) A->row_ptr[C->I[k]]++;

    // 2) prefix sum -> offset
    size_t sum=0;
    for(size_t i=0;i<rows;++i){ 
        size_t c=A->row_ptr[i]; 
        A->row_ptr[i] = sum; 
        sum+=c; 
    }
    A->row_ptr[rows]=sum;
    if (sum != nnz){ 
        fprintf(stderr,"CSR mismatch row_ptr[M]=%zu nnz=%zu\n", A->row_ptr[rows], nnz); 
        return -1; 
    }

    // 3) fill with pointer
    size_t *next = (size_t*)malloc(rows*sizeof(size_t));
    if(!next){ fprintf(stderr,"malloc next\n"); return -1; }
    memcpy(next, A->row_ptr, rows*sizeof(size_t));

    for(size_t k=0;k<nnz;++k){
        const size_t i = C->I[k];
        const size_t p = next[i]++;   // position free for row i
        A->col_idx[p] = C->J[k];
        A->val[p]     = C->V[k];
    }
    free(next);
    return 0;
}

// ---------------- SpMV kernels ----------------
// Sequential SpMV: y = A * x (CSR) 
static void spmv_csr_seq(const csr_t *A, const float *x, float *y){
    for (size_t i=0;i<A->M;++i){
        float s = 0.0f;

        for (size_t p=A->row_ptr[i]; p<A->row_ptr[i+1]; ++p)
            s += A->val[p] * x[A->col_idx[p]];
        y[i] = s;
    }
}
// OpenMP SpMV: y = A * x (CSR) 
static void spmv_csr_omp(const csr_t *A, const float *x, float *y){
    #pragma omp parallel for schedule(runtime)
    for (size_t i=0;i<A->M;++i){
        float s = 0.0f;

        for (size_t p=A->row_ptr[i]; p<A->row_ptr[i+1]; ++p)
            s += A->val[p] * x[A->col_idx[p]];
        y[i] = s;
    }
}
// OpenMP + SIMD SpMV: y = A * x (CSR) 
static void spmv_csr_simd(const csr_t *A, const float *x, float *y){
    #pragma omp parallel for schedule (runtime)
    for (size_t i=0;i<A->M;++i){
        float s = 0.0f;

        const size_t start = A->row_ptr[i];     // aligned hint 
        const size_t end   = A->row_ptr[i+1];   
        #pragma omp simd reduction(+:s)
        for (size_t p=start; p<end; ++p)
            s += A->val[p] * x[A->col_idx[p]];
        y[i] = s;
    }
}

/* ---------------- BENCH ---------------- */

typedef void (*ker_fun_t)(const csr_t*, const float*, float*);

static double bench_ms(ker_fun_t kernel, const csr_t *A, const float *x, float *y,
                       int iters, double *times)
{
    // warm-up
    kernel(A,x,y);
    double best = 1e9;
    for (int t=0; t<iters; ++t){
        double t0=now_sec();
        kernel(A,x,y);
        double t1=now_sec();
        times[t]=(t1-t0)*1e3;
        if(times[t] < best) best=times[t];
    }
    return best;
}

// Seq vs. parallel with thread
static void bench_scaling(const csr_t *A, const float *x, int iters, const char *name, ker_fun_t kernel){
#ifndef _OPENMP
    printf("%s: compiled without OpenMP, only 1 thread.\n", name);
    return;
#else
    const int thread_list[] = {1,2,4,8,16,32};                  
    const int NT = sizeof(thread_list)/sizeof(thread_list[0]);  // num threads

    // scheduling
    const omp_sched_t schedules[] = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
    const char *schedule_names[] = {"static","dynamic","guided"};
    const int NS = sizeof(schedules)/sizeof(schedules[0]);

    printf("\n--- %s scaling (OpenMP) ---\n", name);

    // buffer output
    float *y=NULL;
    if (posix_memalign((void**)&y, 64, A->M * sizeof(float))){
        fprintf(stderr,"posix_memalign y in bench_scaling\n");
        return;
    }
    const double bytes = (double)A->nnz*(sizeof(float)+sizeof(size_t)+sizeof(float))     // val + col_idx + x
                       + (double)(A->M+1)*sizeof(size_t)                                 // row_ptr
                       + (double)A->M*sizeof(float);                                     // y

    int maxT = omp_get_max_threads();  // <-- CAP THREADS
    for(int s=0; s<NS; ++s){
        printf("\n=== schedule: %s ===\n", schedule_names[s]);
        printf("thr   best[ms]   p90[ms]    GB/s   speedup  eff%%\n");
        double best1 = -1.0;   // take best with 1 thread for i speedup

        // loop over threads
        for (int k=0; k<NT; ++k){
            int T = thread_list[k];
            if (T>maxT) continue;           // <-- CAP THREADS, miss oversubscription
            omp_set_num_threads(T);
            omp_set_schedule(schedules[s], 0);
            // take time
            double *times = malloc(iters*sizeof(double));
            // choose kernel
            double best = bench_ms(kernel, A, x, y, iters, times);
            double p90  = percentile(times, iters, 90.0);
            free(times);

            // bandwidth 
            const double gbps = (bytes/1e9) / (best/1e3);

            // speedup
            if (T==1) best1 = best;
            const double speedup = (best1>0.0) ? (best1/best) : 1.0;
            const double eff     = 100.0 * speedup / (double)T;

            printf("%3d  %9.3f  %8.3f  %6.2f   %7.2f  %5.1f\n", T, best, p90, gbps, speedup, eff);
        }
        printf("\n");
    }
    free(y);
#endif
}

/* ---------------- MAIN ---------------- */
int main(int argc, char **argv){
    // args
    if(argc<2){
        fprintf(stderr,"Usage: %s matrix.mtx [iters]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int iters = 10;
    if(argc>2) {
        iters = atoi(argv[2]); 
        if(iters<1) iters=1;
    }
    
    // matrices
    coo_t C; if (mm_read_coo(path, &C)!=0) return 1;
    coo_sort_and_merge(&C);
    csr_t A; if (coo_to_csr(&C, &A)!=0){ coo_free(&C); return 1; }

    // vectors
    float *x=NULL, *y_seq=NULL, *y_omp=NULL, *y_simd=NULL;
    if (posix_memalign((void**)&x,   64, A.N*sizeof(float)) ||
        posix_memalign((void**)&y_seq, 64, A.M*sizeof(float)) ||
        posix_memalign((void**)&y_omp, 64, A.M*sizeof(float)) ||
        posix_memalign((void**)&y_simd, 64, A.M*sizeof(float))) {
        fprintf(stderr,"posix_memalign vec\n"); 
        coo_free(&C); csr_free(&A);
        return 1;
    }

    // init
    srand(12345);
    fill_random(x, A.N);

    // NUMA first touch
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (size_t i=0;i<A.M;++i) y_seq[i] = y_omp[i] = y_simd[i] = 0.0f;
#else
    for (size_t i=0;i<A.M;++i) y_seq[i] = y_omp[i] = y_simd[i] = 0.0f;
#endif  
    
    // benchmarks(ms)
    double *t_seq = (double*)malloc(iters*sizeof(double));
    double *t_omp = (double*)malloc(iters*sizeof(double));
    double *t_simd = (double*)malloc(iters*sizeof(double));
    if(!t_seq || !t_omp || !t_simd){
        fprintf(stderr,"malloc times\n");
        coo_free(&C); csr_free(&A); free(t_seq); free(t_omp); free(t_simd); free(x); free(y_seq); free(y_omp); free(y_simd);
        return 1;
    }

    double best_seq = bench_ms(spmv_csr_seq, &A, x, y_seq, iters, t_seq);
    double best_omp = bench_ms(spmv_csr_omp, &A, x, y_omp, iters, t_omp);
    double best_simd = bench_ms(spmv_csr_simd, &A, x, y_simd, iters, t_simd);

    double p90_seq = percentile(t_seq, iters, 90.0);
    double p90_omp = percentile(t_omp, iters, 90.0);
    double p90_simd = percentile(t_simd, iters, 90.0);

    float err_omp = max_abs_err(y_seq, y_omp, A.M);
    float err_simd = max_abs_err(y_seq, y_simd, A.M);

    //simple bandwith model
    const double bytes = (double)A.nnz*(sizeof(float)+sizeof(size_t)+sizeof(float))
                   + (double)(A.M+1)*sizeof(size_t)
                   + (double)A.M*sizeof(float);

    const double gb = bytes / 1e9;
    const double ms = 1e3;

    // Check sparsity
    double total = (double)A.M * (double)A.N;
    double density = (double)A.nnz / total;
    double sparsity = 1.0 - density;

    printf("Matrix Market: %s | M=%zu N=%zu nnz=%zu iters=%d\n",path, A.M, A.N, A.nnz, iters);
    printf("  -> Density  = %.6f (%.4f%%)\n", density, density*100.0);
    printf("  -> Sparsity = %.6f (%.4f%%)\n", sparsity, sparsity*100.0);

    printf("%-12s: best=%10.6f ms| p90=%10.6f ms| GB/s=%8.2f | max|err|=%.3g\n","CSR seq",  best_seq,  p90_seq,  gb/(best_seq/ms), 0.0f);
    printf("%-12s: best=%8.3f ms | p90=%8.3f ms | GB/s=%6.2f | max|err|=%.3g\n","CSR omp",  best_omp,  p90_omp,  gb/(best_omp/ms),  err_omp);
    printf("%-12s: best=%8.3f ms | p90=%8.3f ms | GB/s=%6.2f | max|err|=%.3g\n","CSR omp+simd", best_simd, p90_simd, gb/(best_simd/ms), err_simd);

#ifdef _OPENMP
    // scaling analysis
    bench_scaling(&A, x, iters, "CSR omp", spmv_csr_omp);
    bench_scaling(&A, x, iters, "CSR omp+simd", spmv_csr_simd);
#endif

    // free
    free(t_seq); free(t_omp); free(t_simd);
    free(x); free(y_seq); free(y_omp); free(y_simd);
    coo_free(&C); csr_free(&A);

    return 0;
}
