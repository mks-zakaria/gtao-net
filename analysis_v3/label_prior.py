"""Label-only reference: variance shares and the leave-person-out experiment prior.
Usage: python analysis_v3/label_prior.py corpus_full.npz   (no model, no features)."""
import numpy as np, sys
def k(a,b,weights=None,n=3):
    O=np.zeros((n,n))
    for i,j in zip(a,b): O[i,j]+=1
    W=np.array([[(i-j)**2 for j in range(n)] for i in range(n)])/(n-1)**2
    E=np.outer(O.sum(1),O.sum(0))/O.sum(); return 1-(W*O).sum()/(W*E).sum()
d=np.load(sys.argv[1]); y=d['y'].astype(float); s=d['subjects']
grp=np.array([x.split('/')[0] for x in s]); exp=np.array([x.split('/')[1] for x in s])
per=np.array([x.split('/')[0]+'/'+x.split('/')[2] for x in s]); ge=np.char.add(np.char.add(grp,'/'),exp)
def share(lab):
    m={u:y[lab==u].mean() for u in np.unique(lab)}; mu=np.array([m[u] for u in lab]); return ((mu-y.mean())**2).sum()/((y-y.mean())**2).sum()
print('label-variance share: person %.2f | experiment(group-exp) %.2f | experiment index %.2f | session %.2f'%(share(per),share(ge),share(exp),share(s)))
# leave-person-out experiment prior: predict mean label of OTHER persons in same group/experiment
pred=np.empty_like(y)
for u in np.unique(ge):
    idx=np.where(ge==u)[0]
    for p in np.unique(per[idx]):
        a=idx[per[idx]==p]; b=idx[per[idx]!=p]; pred[a]=y[b].mean()
# map continuous prior to classes via thresholds matching marginal distribution
cut=np.quantile(pred,[ (y==0).mean(), (y<=1).mean()]); pc=np.digitize(pred,cut,right=True)
print('experiment-prior (leave-person-out) QWK %.3f  pearson r %.3f'%(k(y.astype(int),pc,weights='quadratic'),np.corrcoef(pred,y)[0,1]))
# within-session label variability
w=[y[s==u].std() for u in np.unique(s)]; print('mean within-session label s.d. %.3f; sessions with constant label %d/%d'%(np.mean(w),sum(np.array(w)==0),len(w)))

# --- class-moment prior: mean label of the OTHER students at the same 4 s moment of the same class
t=np.array([float(x.split('#')[1]) for x in d['keys']])
t0={u:t[ge==u].min() for u in np.unique(ge)}
slot=np.array([int(round((t[i]-t0[ge[i]])/4.0)) for i in range(len(t))])
cm=np.char.add(np.char.add(ge,'#'),slot.astype(str))
from collections import defaultdict
S=defaultdict(float); N=defaultdict(int)
for c,v in zip(cm,y): S[c]+=v; N[c]+=1
ok=np.array([N[c]>1 for c in cm]); pm=np.array([(S[c]-v)/(N[c]-1) if N[c]>1 else np.nan for c,v in zip(cm,y)])
cut=np.quantile(pm[ok],[(y[ok]==0).mean(),(y[ok]<=1).mean()]); pc=np.digitize(pm[ok],cut,right=True)
print('class-moment prior (other students, same moment) QWK %.3f  r %.3f  n=%d'%(k(y[ok].astype(int),pc),np.corrcoef(pm[ok],y[ok])[0,1],ok.sum()))
print('label-variance share: class-moment %.2f'%share(cm))
