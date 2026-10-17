/* Frequency-labelled RX prefix area. No fast-math, clipping, or fixed-grid shortcut. */
#include <math.h>
#include <complex.h>
#include <stdint.h>
#define MAXTOK 128
static int response(int nt,const int32_t *op,const int32_t *arg,const double *p,
                    double f,double complex *out,double complex *der){
    if(nt<0){
        double w=6.283185307179586476925286766559*f;
        double complex z=0.0,d=0.0;
        for(int j=0;j<6;j++){
            double conductance=1.0/p[3*j],wl=w*p[3*j+1],wc=w*p[3*j+2];
            double b=wc-1.0/wl,den=conductance*conductance+b*b;
            double complex v=(conductance-I*b)/den;
            z+=v;d+=-I*(wc+1.0/wl)*v*v;
        }
        *out=z;*der=d;
        return (!isfinite(creal(z))||!isfinite(cimag(z)))?3:0;
    }
    double complex z[MAXTOK],d[MAXTOK];int top=0;
    double w=6.283185307179586476925286766559*f;
    for(int i=0;i<nt;i++){
        int k=op[i],a=arg[i];
        if(k<10){
            double complex v=k==1?p[a]:(k==2?I*w*p[a]:1.0/(I*w*p[a]));
            z[top]=v;d[top]=(k==1?0.0:(k==2?v:-v));top++;
        }else{
            int start=top-a;double complex v=0.0,dd=0.0;
            if(start<0)return 2;
            for(int j=start;j<top;j++){
                if(k==10){v+=z[j];dd+=d[j];}
                else{v+=1.0/z[j];dd-=d[j]/(z[j]*z[j]);}
            }
            if(k==11){dd=-dd/(v*v);v=1.0/v;}
            top=start;z[top]=v;d[top]=dd;top++;
        }
    }
    if(top!=1||!isfinite(creal(z[0]))||!isfinite(cimag(z[0])))return 3;
    *out=z[0];*der=d[0];return 0;
}
int model(int nt,const int32_t *op,const int32_t *arg,const double *p,int nf,
          const double *f,double complex *z,double complex *d){
    for(int i=0;i<nf;i++){int rc=response(nt,op,arg,p,f[i],z+i,d+i);if(rc)return rc;}return 0;
}
int integrate(int nt,const int32_t *op,const int32_t *arg,const double *p,const double *ref,
              int nw,const double *lo,const double *span,const double *scale,
              const int32_t *offset,const double *edges,int n,const double *u,
              const double *weights,const double *anti,double *out,int64_t *queries,int flags){
    if(n>64||nt>MAXTOK)return 4;
    *queries=0;
    for(int k=0;k<nw;k++){
        double complex z0,r0,dummy;
        if(response(nt,op,arg,p,pow(10.0,lo[k]),&z0,&dummy))return 5;
        if(response(nt,op,arg,ref,pow(10.0,lo[k]),&r0,&dummy))return 5;
        z0/=scale[k];r0/=scale[k];*queries+=2;
        double prefix=0.0,loss=0.0,point=0.0,length=0.0;
        for(int panel=offset[k];panel<offset[k+1]-1;panel++){
            double left=edges[panel],width=edges[panel+1]-left;
            double g[64];
            for(int j=0;j<n;j++){
                double t=left+width*u[j],f=pow(10.0,lo[k]+span[k]*t);
                double complex z,r,dz,dr;
                if(response(nt,op,arg,p,f,&z,&dz))return 6;
                if(response(nt,op,arg,ref,f,&r,&dr))return 6;
                z/=scale[k];r/=scale[k];
                dz*=2.30258509299404568402*span[k]/scale[k];
                dr*=2.30258509299404568402*span[k]/scale[k];
                if(flags&1)g[j]=0.5*cimag(conj(z-z0)*dz-conj(r-r0)*dr);
                if(flags&4){double er=cabs(z-r);point+=width*weights[j]*er*er;}
                if(flags&2)length+=width*weights[j]*cabs(dr);
            }
            double full=0.0;
            if(flags&1)for(int j=0;j<n;j++){
                double a=prefix;
                for(int m=0;m<n;m++)a+=width*anti[j*n+m]*g[m];
                loss+=width*weights[j]*a*a;
                full+=width*weights[j]*g[j];
            }
            prefix+=full;*queries+=2*n;
        }
        if(!isfinite(loss)||!isfinite(point)||!isfinite(length))return 7;
        out[3*k]=loss;out[3*k+1]=point;out[3*k+2]=length;
    }
    return 0;
}
