"""Two-stage LD then independent HA decoding, using the existing cost scale."""
import numpy as np
import pandas as pd


def _two_state(emissions,costs):
    m=len(emissions)
    if not m:return np.zeros(0,np.uint8)
    back=np.zeros((m,2),np.uint8);dp=emissions[0].copy()
    for i in range(1,m):
        candidates=dp[:,None]-costs[i]
        best=np.argmax(candidates,axis=0);back[i]=best
        dp=emissions[i]+candidates[best,np.arange(2)]
    states=np.zeros(m,np.uint8);states[-1]=np.argmax(dp)
    for i in range(m-1,0,-1):states[i-1]=back[i,states[i]]
    return states


def decode_hierarchical(scores,half,turnover,stop,restart,cfg):
    from .temporal_decoder import DecodeResult,state_cost_matrix,transition_costs,_soft_short_run_refine,_segments
    score=np.asarray(scores,float);half=np.asarray(half,int);n=len(half)
    if score.shape!=(n,3) or not np.isfinite(score).all():raise ValueError('Invalid score matrix')
    hints=[np.zeros(n) if a is None else np.asarray(a,float) for a in (turnover,stop,restart)]
    if any(a.shape!=(n,) or not np.isfinite(a).all() for a in hints):raise ValueError('Invalid hints')
    base=state_cost_matrix(cfg);labels=np.full(n,'X',dtype='<U1');dead=np.zeros(n,bool);segment_id=np.full(n,-1,int)
    removed=0;serial=0
    for h in (1,2):
        ix=np.flatnonzero(half==h)
        if not len(ix):continue
        if np.any(np.diff(ix)!=1):raise ValueError('Each half must be contiguous')
        all_costs=np.stack([transition_costs(base,*(float(a[i]) for a in hints),cfg) for i in ix])
        binary=np.column_stack([score[ix,:2].max(axis=1),score[ix,2]])
        ld_cost=np.zeros((len(ix),2,2));ld_cost[:,0,1]=all_costs[:,0,2];ld_cost[:,1,0]=all_costs[:,2,0]
        state=_two_state(binary,ld_cost).astype(bool);dead[ix]=state;labels[ix[state]]='D'
        for left,right,code in _segments(state.astype(np.uint8)):
            if code:continue
            ids=ix[left:right];ha_cost=np.zeros((len(ids),2,2))
            ha_cost[:,0,1]=all_costs[left:right,0,1];ha_cost[:,1,0]=all_costs[left:right,1,0]
            teams=_two_state(score[ids,:2],ha_cost)
            teams,count=_soft_short_run_refine(teams,score[ids],cfg);removed+=count
            labels[ids]=np.array(['H','A'])[teams];segment_id[ids]=serial;serial+=1
    transitions=[]
    changes=np.flatnonzero((labels[1:]!=labels[:-1])&(half[1:]==half[:-1])&(half[1:]!=0))+1
    for i in changes:transitions.append({'half':int(half[i]),'frame':int(i),'from':labels[i-1],'to':labels[i]})
    game_state=pd.DataFrame({'frame':np.arange(n),'half':half,'dead_decoded':dead,'live_segment_id':segment_id,
        'dead_minus_best_live':score[:,2]-score[:,:2].max(axis=1)})
    return DecodeResult(labels,pd.DataFrame(transitions,columns=['half','frame','from','to']),
        {'short_runs_merged':removed,'transition_count':len(transitions),'method':'hierarchical','live_segments':serial,
         'semantics':'LD uses same HD/DH costs and hints; HA resets per live interval with existing HA cost; no duration filter on D'},game_state)
