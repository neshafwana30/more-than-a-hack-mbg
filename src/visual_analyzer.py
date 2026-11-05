"""Optional local tactical-video ball recovery with conservative quality gating.

Requires a USER-SUPPLIED local Ultralytics YOLO weights file trained to detect
COCO sports ball (class 32). No automatic download or cloud inference. No model
is bundled. A missing optional dependency/weights is a neutral no-op, not fake
visual evidence. The recovered ball is approximate pitch-plane position only.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
import pandas as pd


@dataclass
class VisualResult:
    observations: pd.DataFrame
    diagnostics: dict


def _empty(reason):
    return VisualResult(pd.DataFrame(columns=['frame','x','y','quality']),{'status':'skipped','reason':reason,'accepted':0})


def project_pitch_to_image(matrix, x, y):
    H=np.asarray(matrix,float).reshape(3,3)
    if not np.isfinite(H).all():return None
    p=H @ np.array([x,y,1.0])
    if not np.isfinite(p).all() or abs(p[2])<1e-9:return None
    return float(p[0]/p[2]),float(p[1]/p[2])


def image_to_pitch(matrix, pixel_x, pixel_y):
    H=np.asarray(matrix,float).reshape(3,3)
    if not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-10:return None
    p=np.linalg.solve(H,np.array([pixel_x,pixel_y,1.0]))
    if not np.isfinite(p).all() or abs(p[2])<1e-9:return None
    return float(p[0]/p[2]),float(p[1]/p[2])


def _crop_box(frame, coords, H, padding):
    height,width=frame.shape[:2]
    if coords is None or H is None:return (0,0,width,height)
    loc=project_pitch_to_image(H,*coords)
    if loc is None:return (0,0,width,height)
    x,y=loc
    return max(0,int(x-padding)),max(0,int(y-padding)),min(width,int(x+padding)),min(height,int(y+padding))


def _near_track_ball(features:pd.DataFrame, t:int, limit:int=20):
    n=len(features); half=int(features.at[t,'half'])
    if half==0:return None
    for dt in range(limit+1):
        for j in (t-dt,t+dt):
            if 0<=j<n and int(features.at[j,'half'])==half and bool(features.at[j,'ball_ok']):
                return float(features.at[j,'ball_x']),float(features.at[j,'ball_y'])
    return None


def analyze_video_windows(windows:pd.DataFrame, features:pd.DataFrame, video_path:str,
                          homography_path:str, config:dict|None=None)->VisualResult:
    cfg=config or {}
    if len(windows)==0:return _empty('no_selected_windows')
    weights=cfg.get('local_yolo_weights')
    if not weights or not Path(str(weights)).is_file():return _empty('local_detector_weights_not_configured')
    if not Path(video_path).is_file() or not Path(homography_path).is_file():
        return _empty('video_or_homography_unavailable')
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        return _empty('install_opencv_and_ultralytics_for_optional_video')
    try:
        model=YOLO(str(weights))
        hg=pd.read_csv(homography_path)
        if len(hg)!=len(features) or not {'frame','valid','reprojection_error'}.issubset(hg.columns):
            return _empty('homography_shape_mismatch')
        matrix_cols=[f'h{i}{j}' for i in range(3) for j in range(3)]
        if not set(matrix_cols).issubset(hg.columns):return _empty('homography_columns_missing')
        if not np.array_equal(hg.frame.to_numpy(int),features.frame.to_numpy(int)):
            return _empty('homography_frame_mismatch')
        matrices=hg[matrix_cols].to_numpy(float).reshape(-1,3,3)
        valid=hg.valid.to_numpy(int)>0
        err=hg.reprojection_error.to_numpy(float)
        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened():return _empty('video_decode_failed')
        candidates=[]
        stride=max(1,int(cfg.get('video_frame_stride',2)))
        max_err=float(cfg.get('max_reprojection_error',2.0))
        pad=int(cfg.get('crop_padding_px',180))
        min_conf=float(cfg.get('detector_min_confidence',0.35))
        try:
            for window in windows.itertuples(index=False):
                for t in range(int(window.start_frame),int(window.end_frame)+1,stride):
                    if not valid[t] or not np.isfinite(err[t]) or err[t]>max_err or not np.isfinite(matrices[t]).all():
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES,t)
                    ok,frame=cap.read()
                    if not ok:continue
                    coords=_near_track_ball(features,t,limit=int(cfg.get('crop_position_age_frames',20)))
                    x0,y0,x1,y1=_crop_box(frame,coords,matrices[t],pad)
                    if x1<=x0 or y1<=y0:continue
                    crop=frame[y0:y1,x0:x1]
                    result=model.predict(source=crop,classes=[32],conf=min_conf,verbose=False)[0]
                    if result.boxes is None or len(result.boxes)==0:continue
                    confidence=result.boxes.conf.cpu().numpy()
                    best=int(np.argmax(confidence))
                    xyxy=result.boxes.xyxy[best].cpu().numpy().astype(float)
                    u=float((xyxy[0]+xyxy[2])/2+x0)
                    v=float((xyxy[1]+xyxy[3])/2+y0)
                    pos=image_to_pitch(matrices[t],u,v)
                    if pos is None or not(-0.03<=pos[0]<=1.03 and -0.03<=pos[1]<=1.03):continue
                    candidates.append((t,float(pos[0]),float(pos[1]),float(confidence[best])))
        finally:
            cap.release()
        # Spatial and temporal persistence: isolated sports-ball detections are rejected.
        candidates.sort();accepted=[]
        max_speed=float(cfg.get('max_video_ball_speed_mps',45.0))
        fps=float(cfg.get('fps',25))
        for i,(t,x,y,conf) in enumerate(candidates):
            neighbors=[]
            for j in (i-1,i+1):
                if 0<=j<len(candidates):
                    tt,xx,yy,cc=candidates[j]
                    delta=abs(t-tt)
                    if 0<delta<=max(6,3*stride):
                        mps=np.hypot((x-xx)*105.3,(y-yy)*68.0)*fps/delta
                        if mps<=max_speed:neighbors.append(cc)
            if len(neighbors)>0 and conf>=min_conf:
                accepted.append({'frame':t,'x':x,'y':y,'quality':min(1.0,conf*(0.65+0.35*min(1.0,len(neighbors))))})
        obs=pd.DataFrame(accepted,columns=['frame','x','y','quality'])
        return VisualResult(obs,{'status':'completed','raw_candidates':len(candidates),'accepted':len(obs),'windows':len(windows)})
    except Exception as ex:
        warnings.warn(f'Optional video analysis failed; no visual evidence used: {type(ex).__name__}: {ex}',RuntimeWarning)
        return _empty(f'video_failed:{type(ex).__name__}')
