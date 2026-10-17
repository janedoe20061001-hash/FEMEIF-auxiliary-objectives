"""Set-valued valley occupancy and strict translation-signature evidence.

This is a diagnostic, not a general deformation matcher or an objective.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/valley_perturbation_20261017'))
from physical import *
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/valley_attribution_20261017'
POSITION_TOL=1e-9

def location(x,t):
    lo,hi=t['core']
    if abs(x-lo)<=POSITION_TOL:return 'left_boundary'
    if abs(x-hi)<=POSITION_TOL:return 'right_boundary'
    return 'in_core' if lo<x<hi else 'outside_core'

def signature(c,ref,root,t):
    width=t['core'][1]-t['core'][0];center=t['center'];offsets=np.unique(np.r_[np.linspace(t['core'][0]-center,t['core'][1]-center,17),0.])
    z,v=rx(c,root['x']+offsets);r,vr=rx(ref,center+offsets);z/=t['scale'];r/=t['scale'];v*=width/t['scale'];vr*=width/t['scale']
    ez=abs(z-r);ev=abs(v-vr);zok=np.all(ez<=1e-8+1e-7*abs(r));vok=np.all(ev<=1e-8+1e-7*abs(vr))
    return dict(max_normalized_state_residual=float(max(ez)),mean_normalized_state_residual=float(np.mean(ez)),max_normalized_derivative_residual=float(max(ev)),mean_normalized_derivative_residual=float(np.mean(ev)),state_numerically_consistent=bool(zok),derivative_numerically_consistent=bool(vok),translation_signature_consistent=bool(zok and vok),signature_points=len(offsets))

def level(model,target_record,n):
    # No family, perturbation amount, expected center, source label, or seed input.
    c=Physical(model);ref=Physical(target_record['model']);t=target_record;width=t['core'][1]-t['core'][0];lo=t['center']-4*width;hi=t['center']+4*width;x=grid(c,ref,lo,hi,n)
    x=np.unique(np.r_[x,t['core'],t['center']]);rr=roots(SimpleNamespace(c=c),c.p,x);valleys=[r for r in rr if r['kind']=='valley'];z,_=rx(c,x);L=float(sum(abs(np.diff(z))));zero=L<=1e-12*t['scale']
    candidates=[]
    for i,r in enumerate(valleys):
        candidates.append(dict(index=i,root=r,location=location(r['x'],t),frequency_offset_decades=float(r['x']-t['center']),frequency_offset_core_widths=float((r['x']-t['center'])/width),**signature(c,ref,r,t)))
    inside=[r for r in candidates if r['location']=='in_core'];boundary=[r for r in candidates if r['location'].endswith('boundary')];consistent=[r for r in candidates if r['translation_signature_consistent']]
    occupancy='zero_motion' if zero else 'no_valley_in_search' if not candidates else 'boundary_only' if not inside and boundary else 'no_valley_in_core' if not inside else 'one_core_candidate' if len(inside)==1 else 'multiple_core_candidates'
    relation='no_strict_translation_evidence' if not consistent else 'one_signature_consistent_candidate' if len(consistent)==1 else 'multiple_signature_consistent_candidates'
    return dict(n=n,search=[lo,hi],core=t['core'],path_length_in_search=L,zero_motion=bool(zero),all_radial_extrema=rr,candidates=candidates,core_valley_count=len(inside),boundary_valley_count=len(boundary),search_valley_count=len(candidates),occupancy=occupancy,signature_relation=relation,signature_consistent_count=len(consistent),signature_center_location=consistent[0]['location'] if len(consistent)==1 else None,signature_center_x=consistent[0]['root']['x'] if len(consistent)==1 else None)

def stable(a,b):
    ra,rb=a['all_radial_extrema'],b['all_radial_extrema']
    roots_ok=len(ra)==len(rb) and all(u['kind']==v['kind'] and abs(u['x']-v['x'])<=POSITION_TOL for u,v in zip(ra,rb))
    ca,cb=a['candidates'],b['candidates']
    candidates_ok=len(ca)==len(cb) and all(u['location']==v['location'] and u['translation_signature_consistent']==v['translation_signature_consistent'] for u,v in zip(ca,cb))
    categories_ok=all(a[key]==b[key] for key in ['zero_motion','occupancy','signature_relation','signature_center_location'])
    return dict(root_count_type_location=roots_ok,candidate_location_signature=candidates_ok,categories=categories_ok)
