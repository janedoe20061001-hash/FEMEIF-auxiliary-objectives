#include <stdint.h>
#include <math.h>

/* Exact same arithmetic as unit-vector squared difference, not an approximation.
 * Terms are returned in column-major order so NumPy retains the original
 * frequency-pair summation layout. No fast-math or multiply-add contraction. */
void direction_terms(const double *p, int64_t batch, int64_t n, int64_t pairs,
                     const int64_t *ii, const int64_t *jj,
                     const double *target, const double *weights, double *out) {
    for (int64_t k=0;k<pairs;k++) {
        const int64_t i=ii[k], j=jj[k];
        const double t0=target[3*k],t1=target[3*k+1],t2=target[3*k+2];
        for (int64_t r=0;r<batch;r++) {
            const double *a=p+3*(r*n+i), *b=p+3*(r*n+j);
            const double d0=b[0]-a[0],d1=b[1]-a[1],d2=b[2]-a[2];
            const double length=sqrt((d0*d0+d1*d1)+d2*d2);
            const double e0=d0/length-t0,e1=d1/length-t1,e2=d2/length-t2;
            out[r+batch*k]=((e0*e0+e1*e1)+e2*e2)*weights[k];
        }
    }
}
