#include <math.h>
#include <complex.h>
#include <stdint.h>

typedef struct { double a,p,t,u; } V;
typedef struct { const double *p,*target; int points,depth,error; double ma,mp,density; } State;
static V value(double x,const double *p,const double *target){
    double w=2.0*3.14159265358979323846*pow(10.0,x);
    double complex s=I*w,z=0.0,t=0.0;
    for(int j=0;j<6;j++){
        double r=p[3*j],l=p[3*j+1],c=p[3*j+2];
        z+=r*s*l/(r+s*l+r*s*s*l*c);
        r=target[3*j];l=target[3*j+1];c=target[3*j+2];
        t+=r*s*l/(r+s*l+r*s*s*l*c);
    }
    V v={20.0*log10(cabs(z)),atan2(cimag(z),creal(z)),20.0*log10(cabs(t)),atan2(cimag(t),creal(t))};return v;
}
static void interval(State *st,double a,double b,V va,V vb,int level){
    if(st->error)return;
    if(level>32){st->error=1;return;}
    if(level>st->depth)st->depth=level;
    double m=(a+b)/2.0;
    V vm=value(m,st->p,st->target);st->points++;
    if(st->points>10000){st->error=3;return;}
    double tol=0.5*3.14159265358979323846/180.0;
    int fail=fabs(vm.a-(va.a+vb.a)/2.0)>0.1||fabs(vm.t-(va.t+vb.t)/2.0)>0.1||fabs(vm.p-(va.p+vb.p)/2.0)>tol||fabs(vm.u-(va.u+vb.u)/2.0)>tol;
    if(fail){interval(st,a,m,va,vm,level+1);interval(st,m,b,vm,vb,level+1);return;}
    double aa=va.a-va.t,am=vm.a-vm.t,ab=vb.a-vb.t;
    double pa=va.p-va.u,pm=vm.p-vm.u,pb=vb.p-vb.u;
    double wl=(m-a)*st->density/2.0,wr=(b-m)*st->density/2.0;
    st->ma+=wl*(aa*aa+am*am);st->ma+=wr*(am*am+ab*ab);
    st->mp+=wl*(pa*pa+pm*pm);st->mp+=wr*(pm*pm+pb*pb);
}
int adaptive_batch(int n,int k,const double *parameters,const double *target,const double *grid,const double *edges,int cells,double *out,int64_t *counts,int64_t *depths){
    for(int i=0;i<n;i++){
        State st={parameters+18*i,target,0,0,0,0.0,0.0,0.0};
        const double *xx=grid+k*i;double a=xx[0];V va=value(a,st.p,target);st.points=1;
        for(int j=1;j<k;j++){
            double b=xx[j];if(b==a)continue;V vb=value(b,st.p,target);st.points++;
            double mid=(a+b)/2.0;int lo=0,hi=cells;
            while(lo<hi){int q=(lo+hi+1)/2;if(edges[q]<=mid)lo=q;else hi=q-1;}
            int cell=lo>=cells?cells-1:lo;st.density=1.0/(cells*(edges[cell+1]-edges[cell]));
            interval(&st,a,b,va,vb,1);if(st.error)return st.error;
            a=b;va=vb;
        }
        if(!isfinite(st.ma)||!isfinite(st.mp))return 4;
        out[2*i]=st.ma;out[2*i+1]=st.mp;counts[i]=st.points;depths[i]=st.depth;
    }return 0;
}
