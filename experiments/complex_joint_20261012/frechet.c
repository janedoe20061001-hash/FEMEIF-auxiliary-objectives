#include <math.h>
#include <stdlib.h>
/* Complex buffers are interleaved real/imaginary doubles, not cast to C complex. */
int batch(int b,int n,const double*t,const double*z,const double*r,double*out,double eta){
 double *prev=malloc(n*sizeof(double)),*cur=malloc(n*sizeof(double));
 if(!prev||!cur){free(prev);free(cur);return 1;}
 for(int a=0;a<b;a++){
  for(int j=0;j<n;j++)prev[j]=INFINITY;
  for(int i=0;i<n;i++){
   for(int j=0;j<n;j++)cur[j]=INFINITY;
   for(int j=0;j<n;j++){
    double dt=t[i]-t[j];if(fabs(dt)>eta+1e-12)continue;
    double re=z[2*(a*n+i)]-r[2*j],im=z[2*(a*n+i)+1]-r[2*j+1];
    double d=dt*dt+re*re+im*im,v=prev[j];
    if(i==0&&j==0)cur[j]=d;
    else{if(j)v=fmin(v,cur[j-1]);if(i&&j)v=fmin(v,prev[j-1]);cur[j]=fmax(v,d);}
   }
   double*tmp=prev;prev=cur;cur=tmp;
  }
  out[a]=sqrt(prev[n-1]);if(!isfinite(out[a])){free(prev);free(cur);return 2;}
 }
 free(prev);free(cur);return 0;
}
