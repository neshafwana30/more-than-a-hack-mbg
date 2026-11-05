import numpy as np
import pandas as pd
import pytest
from src.submission import save_submission,validate_submission_file,segments_for_labels


def test_csv_bytes_reproducible(tmp_path):
    labels=np.array(list('XXHHDDHAAAX'))
    half=np.array([0,0,1,1,1,1,1,2,2,2,0])
    p=tmp_path/'first.csv';q=tmp_path/'second.csv'
    a=save_submission(labels,p,expected_count=len(labels),half=half)
    b=save_submission(labels,q,expected_count=len(labels),half=half)
    assert a['sha256']==b['sha256']
    assert p.read_text().startswith('frame,label\n0,X\n')
    assert validate_submission_file(p,len(labels))
    seg=segments_for_labels(labels,half)
    assert len(seg)==4


def test_reject_blank_or_graded_x(tmp_path):
    with pytest.raises(ValueError):save_submission(np.array(['H','X']),tmp_path/'bad.csv',2,np.array([1,1]))
