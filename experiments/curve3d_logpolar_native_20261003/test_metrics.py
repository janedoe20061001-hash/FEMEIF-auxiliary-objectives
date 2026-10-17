"""Software/formula tests only; no reduced-budget optimization."""
import numpy as np
from global_metrics import GlobalObjectives,AUX

f=np.geomspace(10,1e6,15);u=np.linspace(0,1,len(f))
z=10**(1+u*u)*np.exp(1j*(-.6+u))
obj=GlobalObjectives(f,aux=AUX,target_z=z)
np.testing.assert_allclose(obj.evaluate_spectra(z),0,atol=1e-14)
candidate=np.array([z*2,z*np.exp(.2j),z[::-1],10**(1+.3*u)*np.exp(.1j*u)])
value=obj.evaluate_spectra(candidate)
np.testing.assert_array_equal(value[:,:2],GlobalObjectives(f,target_z=z).evaluate_spectra(candidate))
p=obj.path(candidate)
np.testing.assert_allclose(obj.orientation_loss(p,1),obj.orientation_loss(p,32),rtol=1e-14,atol=1e-14)
for row in range(len(candidate)):
    numerator=denominator=0.
    w=np.r_[obj.du[0]/2,(obj.du[:-1]+obj.du[1:])/2,obj.du[-1]/2]
    for i in range(len(f)):
        for j in range(i+1,len(f)):
            a=p[row,j]-p[row,i];b=obj.p[j]-obj.p[i]
            cosine=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
            weight=w[i]*w[j];numerator+=weight*2*(1-cosine);denominator+=weight
    np.testing.assert_allclose(value[row,-1],numerator/denominator,rtol=1e-12,atol=1e-13)
assert (value[:,-1]>=0).all() and (value[:,-1]<=4+1e-13).all()
# Pure amplitude/phase translations leave chord directions unchanged; the
# historical primary objectives and arc-position discrepancy still detect them.
np.testing.assert_allclose(value[:2,-1],0,atol=1e-25)
assert (value[:2,:2].sum(axis=1)>0).all()
try:obj.evaluate_spectra(np.zeros(len(f),complex))
except ValueError:pass
else:raise AssertionError('Log-domain error should be explicit')
print('PASS: identity, historical primary equality, independently computed all-pair direction loss, numerical chunk invariance, explicit zero-domain handling')
