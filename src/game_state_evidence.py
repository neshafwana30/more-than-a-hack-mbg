"""D2 engineered, quality-aware signed Dead-minus-Live evidence (offline)."""
import numpy as np
import pandas as pd

DEFAULTS={'prior_delta':-.30,'player_window_frames':50,'ball_lag_frames':5,'ball_window_frames':25,
    'minimum_window_fraction':.6,'minimum_player_quality':.65,'walking_speed_mps':1.3,'pace_scale_mps':.5,
    'pace_weight':.30,'still_speed_mps':.5,'still_weight':.35,'fast_speed_mps':5.,'fast_weight':.40,
    'outside_margin_m':.5,'outside_confirm_frames':3,'outside_weight':1.20,'anchor_weight':.30,
    'carry_enabled':False,'carry_max_frames':75}


def _settings(supplied):
    if not isinstance(supplied,dict) or set(supplied)-set(DEFAULTS):raise ValueError('Invalid signed game-state settings')
    c={**DEFAULTS,**supplied}
    for k in ['player_window_frames','ball_lag_frames','ball_window_frames','outside_confirm_frames','carry_max_frames']:
        if isinstance(c[k],bool) or not isinstance(c[k],int) or c[k]<1:raise ValueError(f'{k} must be integer >=1')
    if not isinstance(c['carry_enabled'],bool):raise ValueError('carry_enabled must be boolean')
    for k in set(c)-{'carry_enabled','player_window_frames','ball_lag_frames','ball_window_frames','outside_confirm_frames','carry_max_frames'}:
        if isinstance(c[k],bool) or not np.isfinite(float(c[k])):raise ValueError(f'Invalid {k}')
        if k!='prior_delta' and c[k]<0:raise ValueError(f'{k} must be >=0')
    if not 0<c['minimum_window_fraction']<=1 or not 0<=c['minimum_player_quality']<=1:raise ValueError('Invalid quality fraction')
    if c['pace_scale_mps']<=0 or c['outside_margin_m']<=0 or c['fast_speed_mps']<=c['still_speed_mps']:raise ValueError('Invalid scale/threshold')
    return c


def _smooth(x,width,fraction):
    # Fixed-denominator centred window; missing data never becomes zero motion.
    left=width//2;right=width-1-left;ok=np.isfinite(x)
    kernel=np.ones(width)
    count=np.convolve(np.pad(ok.astype(float),(left,right)),kernel,'valid')
    total=np.convolve(np.pad(np.where(ok,x,0.),(left,right)),kernel,'valid')
    mean=np.divide(total,count,out=np.full(len(x),np.nan),where=count>=width*fraction)
    return mean,count/width


def estimate_signed_game_state(f,q,e,settings,fps=25,pitch=(105.3,68.0)):
    from .live_dead import LiveDeadResult
    c=_settings(settings);n=len(f);fps=float(fps);pitch=np.asarray(pitch,float)
    if not np.isfinite(fps) or fps<=0 or pitch.shape!=(2,) or not np.isfinite(pitch).all() or (pitch<=0).any():raise ValueError('Invalid fps/pitch')
    if not np.array_equal(f.frame,q.frame) or not np.array_equal(f.frame,e.frame):raise ValueError('Evidence frame mismatch')
    half=f.half.to_numpy(int);ids=f.frame.to_numpy(int);ball=f[['ball_x','ball_y']].to_numpy(float)*pitch
    seen=q.ball_observed.to_numpy(bool)&np.isfinite(ball).all(axis=1)
    motion=np.column_stack([f.player_motion_robust_h,f.player_motion_robust_a]).astype(float)
    reliable=np.column_stack([f.centroid_valid_h,f.centroid_valid_a]).astype(bool)&np.isfinite(motion)
    good_players=reliable.all(axis=1)&q.player_quality_weight.ge(c['minimum_player_quality']).to_numpy()
    raw_player=np.where(good_players,motion.mean(axis=1),np.nan)
    sm_player=np.full(n,np.nan);sm_ball=np.full(n,np.nan);ball_lag=np.full(n,np.nan)
    player_fraction=np.zeros(n);ball_fraction=np.zeros(n);outside=np.zeros(n,bool);confirmed=np.zeros(n,bool);held=np.zeros(n);age=np.full(n,-1,int)
    normalized=f[['ball_x','ball_y']].to_numpy(float)
    outside_distance=np.maximum.reduce([-normalized[:,0]*pitch[0],(normalized[:,0]-1)*pitch[0],-normalized[:,1]*pitch[1],(normalized[:,1]-1)*pitch[1]])
    outside=seen&~q.suspected_ball_jump.to_numpy(bool)&(outside_distance>c['outside_margin_m'])
    cuts=np.r_[0,np.flatnonzero((np.diff(half)!=0)|(np.diff(ids)!=1))+1,n]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        if lo==hi or half[lo]==0:continue
        sm_player[lo:hi],player_fraction[lo:hi]=_smooth(raw_player[lo:hi],c['player_window_frames'],c['minimum_window_fraction'])
        lag=c['ball_lag_frames'];length=hi-lo
        if length>lag:
            right=np.arange(lo+lag,hi);left=right-lag
            prefix=np.r_[0,np.cumsum(seen[lo:hi].astype(int))]
            complete=(prefix[right-lo+1]-prefix[left-lo])==lag+1
            speed=np.linalg.norm(ball[right]-ball[left],axis=1)*fps/lag
            ok=complete&np.isfinite(speed)&(speed<=45.)
            ball_lag[right]=np.where(ok,speed,np.nan)
        sm_ball[lo:hi],ball_fraction[lo:hi]=_smooth(ball_lag[lo:hi],c['ball_window_frames'],c['minimum_window_fraction'])
        streak=0;last_out=None
        for i in range(lo,hi):
            streak=streak+1 if outside[i] else 0
            confirmed[i]=streak>=c['outside_confirm_frames']
            if confirmed[i]:last_out=i
            elif seen[i]:last_out=None
            if c['carry_enabled'] and not seen[i] and last_out is not None:
                elapsed=i-last_out
                if elapsed<=c['carry_max_frames']:
                    held[i]=1.-elapsed/(c['carry_max_frames']+1.);age[i]=elapsed
                else:last_out=None
    player_ok=good_players&np.isfinite(sm_player)&(half!=0)
    ball_ok=seen&np.isfinite(ball_lag)&np.isfinite(sm_ball)&~q.suspected_ball_jump.to_numpy(bool)&(half!=0)
    prior=np.full(n,c['prior_delta'],float)
    pace=np.where(player_ok,c['pace_weight']*np.clip((c['walking_speed_mps']-sm_player)/c['pace_scale_mps'],-2,2),0.)
    still=c['still_weight']*(ball_ok&~outside&(sm_ball<c['still_speed_mps']))
    fast=-c['fast_weight']*(ball_ok&~outside&(sm_ball>c['fast_speed_mps']))
    out=c['outside_weight']*confirmed;carry=c['outside_weight']*held
    anchor=c['anchor_weight']*e.stoppage_anchor.to_numpy(float)
    delta=prior+pace+still+fast+out+carry+anchor
    live=-delta/2;dead=delta/2
    frames=pd.DataFrame({'frame':ids,'live_score':live,'dead_score':dead,
                         'player_motion_evidence':np.where(player_ok,sm_player,np.nan),'has_player_motion':player_ok})
    comp=pd.DataFrame({'frame':ids,'prior_delta':prior,'pace_dead':pace,'still_dead':still,'fast_live':fast,
        'outside_dead':out,'carry_dead':carry,'anchor_dead':anchor,'dead_minus_live':delta,
        'raw_player_motion':raw_player,'smoothed_player_motion':sm_player,'player_window_fraction':player_fraction,
        'lagged_ball_speed_mps':ball_lag,'smoothed_ball_speed_mps':sm_ball,'ball_window_fraction':ball_fraction,
        'player_evidence_available':player_ok,'ball_evidence_available':ball_ok,'outside_observed':outside,
        'outside_confirmed':confirmed,'outside_carry_age_frames':age,'outside_carry_support':held})
    return LiveDeadResult(frames,live,dead,comp,{'model':'signed','settings':c,'fps':fps,'pitch_m':pitch.tolist(),
        'signed_additive_columns':['prior_delta','pace_dead','still_dead','fast_live','outside_dead','carry_dead','anchor_dead'],
        'semantics':'engineered uncalibrated signed Dead-minus-Live utility; live=-delta/2, dead=delta/2; not probabilities',
        'smoothing':'centred offline fixed-denominator coverage windows, reset at half/frame gap; current valid measurement required; no missing-to-stationary substitution',
        'carry':'only confirmed observed outside -> immediately following missing frames; linear decay, finite expiry, any visible nonconfirmed observation resets; no half/gap crossing',
        'inactive_legacy_weights':'legacy live_dead weights are not used in signed mode; no D1 or missing-ball bonus',
        'weight_rationale':'prior gap matches legacy -0.30; pace capped +/-0.60; fast -0.40 offsets maximum slow pace plus prior; outside 1.20 exceeds prior plus strongest fast-player evidence; hypotheses fixed before runs',
        'limitations':'slow/quiet live play and stationary/outside tracking noise may still be ambiguous; centred windows can move boundaries; no automatic restart from tag end'})
