"""A1 experimental ball/nearest-player co-motion support, without player identities."""
from __future__ import annotations
import numpy as np
import pandas as pd



def _explain_gate(frames, quality, cfg, side, gate, support, pair, relative_speed):
    """Read-only explanation; flags overlap, primary reason has explicit precedence."""
    n=len(frames);half=frames.half.to_numpy(int);ids=frames.frame.to_numpy(int)
    names=['outside_half','missing_ball','unreliable_ball_motion','low_player_quality',
           'invalid_position','not_near','block_start','previous_observation_unavailable',
           'previous_not_near','relative_speed_high','insufficient_history','insufficient_support']
    flags={name:np.zeros(n,bool) for name in names}
    reason=np.full(n,'disabled',object);status=np.full(n,'disabled',object)
    if cfg['enabled'] and n:
        cont=np.r_[False,(half[1:]==half[:-1])&(ids[1:]==ids[:-1]+1)]
        start=np.maximum.accumulate(np.where(~cont,np.arange(n),0));age=np.arange(n)-start+1
        coverage=quality.player_quality_weight.to_numpy(float)
        weight=quality.ball_weight.to_numpy(float)
        coordinates=frames[['ball_x','ball_y',f'nearest_{side}_x',f'nearest_{side}_y']].to_numpy(float)
        distance=frames[f'nearest_{side}_m'].to_numpy(float)
        flags['outside_half']=half==0
        flags['missing_ball']=~quality.ball_observed.to_numpy(bool)
        flags['unreliable_ball_motion']=~quality.ball_motion_reliable.to_numpy(bool)|~np.isfinite(weight)|(weight<=0)
        flags['low_player_quality']=~np.isfinite(coverage)|(coverage<float(cfg['minimum_player_quality']))
        flags['invalid_position']=~np.isfinite(coordinates).all(axis=1)|~np.isfinite(distance)|(distance<0)
        flags['not_near']=np.isfinite(distance)&(distance>float(cfg['maximum_distance_m']))
        observation_ok=~(flags['outside_half']|flags['missing_ball']|flags['unreliable_ball_motion']|flags['low_player_quality']|flags['invalid_position'])
        previous_observation_ok=np.r_[False,observation_ok[:-1]]
        previous_near=np.r_[False,~flags['not_near'][:-1]]
        flags['block_start']=~cont
        flags['previous_observation_unavailable']=cont&~previous_observation_ok
        flags['previous_not_near']=cont&previous_observation_ok&~previous_near
        flags['relative_speed_high']=pair&np.isfinite(relative_speed)&(relative_speed>float(cfg['maximum_relative_speed_mps']))
        flags['insufficient_history']=age<int(cfg['window_frames'])
        flags['insufficient_support']=support<float(cfg['minimum_support_fraction'])
        reason=np.select([flags[k] for k in names],names,default='supported').astype(object)
        reason[gate>0]='supported'
        # Not supporting the proxy is not proof of absent possession/acquisition.
        status=np.full(n,'unavailable',object)
        status[np.isin(reason,['not_near','previous_not_near','relative_speed_high','insufficient_support'])]='observed_not_supporting_proxy'
        status[reason=='outside_half']='outside_half';status[gate>0]='supported'
        if np.any((gate==0)&(reason=='supported')):
            raise AssertionError('Unexplained zero acquisition gate')
    data={f'fail_{k}_{side}':v for k,v in flags.items()}
    data[f'primary_reason_{side}']=reason;data[f'evidence_status_{side}']=status
    return data


def acquisition_support(frames, quality, config, fps=25, pitch=(105.3,68.0)):
    defaults={'enabled':False,'application':'contract','bonus_weight':0.5,'fallback_unavailable':False,'window_frames':5,'minimum_support_fraction':0.6,
              'maximum_distance_m':2.5,'maximum_relative_speed_mps':3.0,
              'minimum_player_quality':0.65}
    if not isinstance(config,dict):raise ValueError('acquisition must be a mapping')
    if set(config)-set(defaults):raise ValueError('Unknown acquisition settings')
    cfg={**defaults,**config}
    if not isinstance(cfg['enabled'],bool):raise ValueError('acquisition.enabled must be boolean')
    if not isinstance(cfg['fallback_unavailable'],bool):raise ValueError('acquisition.fallback_unavailable must be boolean')
    if cfg['application'] not in ('contract','bonus'):raise ValueError('acquisition.application must be contract or bonus')
    if isinstance(cfg['bonus_weight'],bool) or not np.isfinite(float(cfg['bonus_weight'])) or float(cfg['bonus_weight'])<0:raise ValueError('acquisition.bonus_weight must be finite >=0')
    if cfg['application']=='bonus' and cfg['fallback_unavailable']:raise ValueError('Bonus application cannot combine with unavailable fallback')
    width=cfg['window_frames']
    if isinstance(width,bool) or not isinstance(width,int) or width<2:raise ValueError('acquisition.window_frames must be an integer >=2')
    for key in ('minimum_support_fraction','maximum_distance_m','maximum_relative_speed_mps','minimum_player_quality'):
        if not np.isfinite(float(cfg[key])):raise ValueError(f'acquisition.{key} must be finite')
    if not 0<float(cfg['minimum_support_fraction'])<=1:raise ValueError('Invalid acquisition support fraction')
    if not 0<=float(cfg['minimum_player_quality'])<=1:raise ValueError('Invalid acquisition quality threshold')
    if float(cfg['maximum_distance_m'])<=0 or float(cfg['maximum_relative_speed_mps'])<0:raise ValueError('Invalid acquisition distance/speed')
    pitch=np.asarray(pitch,float);fps=float(fps)
    if pitch.shape!=(2,) or not np.isfinite(pitch).all() or (pitch<=0).any() or not np.isfinite(fps) or fps<=0:
        raise ValueError('Acquisition requires positive finite FPS and pitch dimensions')
    n=len(frames);half=frames.half.to_numpy(int);ids=frames.frame.to_numpy(int)
    parts={'frame':ids};gates=[]
    if cfg['enabled']:
        ball=frames[['ball_x','ball_y']].to_numpy(float)
        coverage=quality.player_quality_weight.to_numpy(float)
        observed=(quality.ball_observed.to_numpy(bool)&quality.ball_motion_reliable.to_numpy(bool)
                  &np.isfinite(coverage)&(coverage>=float(cfg['minimum_player_quality']))
                  &np.isfinite(quality.ball_weight.to_numpy(float))&(quality.ball_weight.to_numpy(float)>0))
    # Each contiguous half/frame block has independent trailing history.
    starts=np.r_[0,np.flatnonzero((np.diff(half)!=0)|(np.diff(ids)!=1))+1]
    ends=np.r_[starts[1:],n]
    for side in ('h','a'):
        relative_speed=np.full(n,np.nan);pair=np.zeros(n,bool);cue=np.zeros(n,bool)
        support=np.zeros(n,float);gate=np.ones(n,float) if not cfg['enabled'] else np.zeros(n,float)
        if cfg['enabled']:
            nearest=frames[[f'nearest_{side}_x',f'nearest_{side}_y']].to_numpy(float)
            relative=(ball-nearest)*pitch
            distance=frames[f'nearest_{side}_m'].to_numpy(float)
            eligible=(observed&(half!=0)&np.isfinite(relative).all(axis=1)&np.isfinite(distance)
                      &(distance>=0)&(distance<=float(cfg['maximum_distance_m'])))
            pair[1:]=eligible[1:]&eligible[:-1]&(half[1:]==half[:-1])&(ids[1:]==ids[:-1]+1)
            delta=np.linalg.norm(np.diff(relative,axis=0),axis=1)*fps
            relative_speed[1:]=np.where(pair[1:],delta,np.nan)
            cue=pair&np.isfinite(relative_speed)&(relative_speed<=float(cfg['maximum_relative_speed_mps']))
            for left,right in zip(starts,ends):
                if left==right or half[left]==0:continue
                ix=np.arange(left,right);r=np.arange(1,len(ix)+1);l=np.maximum(0,r-width)
                prefix=np.r_[0.,np.cumsum(cue[ix].astype(float))]
                support[ix]=(prefix[r]-prefix[l])/width
                active=(r>=width)&cue[ix]&(support[ix]>=float(cfg['minimum_support_fraction']))
                gate[ix]=support[ix]*active
        parts.update({f'pair_valid_{side}':pair,f'relative_speed_{side}_mps':relative_speed,
                      f'acquisition_cue_{side}':cue,f'acquisition_support_{side}':support,
                      f'acquisition_gate_{side}':gate})
        parts.update(_explain_gate(frames,quality,cfg,side,gate,support,pair,relative_speed))
        gates.append(gate)
    return np.column_stack(gates),pd.DataFrame(parts),{
        'acquisition':cfg,'fps':fps,'pitch_m':pitch.tolist(),
        'semantics':'uncalibrated co-motion support, not possession labels or player identities',
        'window':'trailing fixed denominator; full window inside contiguous half/frame block; current cue required',
        'relative_speed':'norm of change in metric ball-minus-nearest-player vector times FPS; both endpoints reliable and near',
        'reason_precedence':['outside_half','missing_ball','unreliable_ball_motion','low_player_quality','invalid_position','not_near','block_start','previous_observation_unavailable','previous_not_near','relative_speed_high','insufficient_history','insufficient_support','supported'],
        'reason_semantics':'failure flags overlap; primary reason is the first failure in precedence, not a causal ranking; unavailable != observed_not_supporting_proxy; neither is a negative possession label',
        'limitations':'nearest player can change; a switch can suppress support; quiet/contested play can support both teams; passes may lose support'}
