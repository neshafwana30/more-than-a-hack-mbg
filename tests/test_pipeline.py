import numpy as np
from src.pipeline import infer_from_features


def test_synthetic_full_inference(fake_bundle, tmp_path):
    result=infer_from_features(fake_bundle,{'fps':25,'video':{'enabled':False}}, storage_config={'root_dir':str(tmp_path)})
    assert len(result.labels)==len(fake_bundle.frames)
    assert set(result.labels).issubset({'H','A','D'})
    assert result.diagnostics['graded_frames']==len(result.labels)
    assert result.diagnostics['video']['status']=='disabled'
    assert result.diagnostics['possession_spells']>=1
