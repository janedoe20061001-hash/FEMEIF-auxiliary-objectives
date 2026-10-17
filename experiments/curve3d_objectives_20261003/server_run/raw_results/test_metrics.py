"""Formula checks only, not reduced-budget fitting experiments."""
import numpy as np
from global_metrics import GlobalObjectives,AUX,covariance,area,srvf,arc_knots,arc_distance

u=np.array([0.,.13,.42,1.]);du=np.diff(u)
p=np.column_stack([u,2*u+1,3*u-2])
slope=np.array([1.,2.,3.])
np.testing.assert_allclose(covariance(p,du),np.outer(slope,slope)/12,atol=1e-14)
np.testing.assert_allclose(covariance(p+np.array([3.,-7.,8.]),du),covariance(p,du),atol=1e-14)
np.testing.assert_allclose(area(p),0,atol=1e-14)
np.testing.assert_allclose(srvf(p,du),np.broadcast_to(slope/np.sqrt(np.linalg.norm(slope)),(3,3)),atol=1e-14)
triangle=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]])
np.testing.assert_allclose(area(triangle),[0,0,.5])
np.testing.assert_allclose(area(triangle[::-1]),-area(triangle))
np.testing.assert_allclose(area(triangle+7),area(triangle))
refined=np.array([[0,0,0],[.3,0,0],[1,0,0],[1,.6,0],[1,1,0]])
np.testing.assert_allclose(arc_distance(refined,triangle,arc_knots(triangle)),0,atol=1e-28)
np.testing.assert_allclose(arc_distance(triangle+[0,0,2],triangle,arc_knots(triangle)),4,atol=1e-14)
f=10**(2+4*u);z=(1+2*u)+1j*(-1+3*u*u)
obj=GlobalObjectives(f,aux=AUX,target_z=z)
np.testing.assert_allclose(obj.evaluate_spectra(z),0,atol=1e-14)
candidate=np.array([z*1.2,z+.2-1j*.4,z[::-1],np.zeros(len(z),complex)])
value=obj.evaluate_spectra(candidate)
assert np.isfinite(value).all() and (value>=0).all()
plain=GlobalObjectives(f,target_z=z).evaluate_spectra(candidate)
np.testing.assert_array_equal(value[:,:2],plain)
np.testing.assert_array_equal(plain[:,0],np.mean((20*np.log10(np.maximum(abs(candidate),1e-300))-20*np.log10(abs(z)))**2,axis=1))
np.testing.assert_array_equal(plain[:,1],np.mean((np.angle(candidate)-np.angle(z))**2,axis=1))
for coord in ['linear','asinh']:
    path=obj.path(candidate,coord);reference=obj.reference[coord]
    # Chord loss verified with an independent explicit double loop.
    numerator=denominator=0.
    for i in range(len(u)):
        for j in range(i+1,len(u)):
            d=np.linalg.norm(path[0,i]-path[0,j]);t=np.linalg.norm(reference['p'][i]-reference['p'][j])
            w=obj.weights[i]*obj.weights[j];numerator+=w*(d-t)**2;denominator+=w*t*t
    index=AUX.index('chords_'+coord)+2
    np.testing.assert_allclose(value[0,index],np.sqrt(numerator/denominator),rtol=1e-13)
    # Covariance/area/SRVF tested above by analytic closed-form examples.
assert np.arcsinh(0)==0 and np.arcsinh(-3)==-np.arcsinh(3)
print('PASS: identity, signed/zero spectra, analytical covariance/area/SRVF, exact arc integral, independent chords, historical primary equality')
