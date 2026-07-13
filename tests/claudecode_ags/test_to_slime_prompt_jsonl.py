# tests/claudecode_ags/test_to_slime_prompt_jsonl.py
import json
from pathlib import Path

from examples.claudecode_ags.data.to_slime_prompt_jsonl import convert_row, convert_file

REQUIRED_EXTRA = ("instance_id", "image", "problem_statement", "FAIL_TO_PASS", "dataset_type")


def test_convert_row_maps_prompt_and_extra_info():
    src = {
        "instance_id": "getmoto__moto-5386",
        "n_resolved": 3,
        "n_repeats": 8,
        "metadata": {
            "instance_id": "getmoto__moto-5386",
            "problem_statement": "Fix hibernation",
            "image": "swebenchdocker.tencentcloudcr.com/swebench/x:latest",
            "FAIL_TO_PASS": ["t1"],
            "PASS_TO_PASS": ["t2"],
            "dataset_type": "swegym",
            "data_source": "swegym",
            "repo": "getmoto/moto",
        },
    }
    out = convert_row(src)
    assert out["prompt"] == [{"role": "user", "content": "Fix hibernation"}]
    assert out["extra_info"]["instance_id"] == "getmoto__moto-5386"
    assert out["extra_info"]["n_resolved"] == 3
    for k in REQUIRED_EXTRA:
        assert k in out["extra_info"] and out["extra_info"][k] not in (None, "")


def test_convert_row_rejects_missing_problem_statement():
    import pytest

    with pytest.raises(ValueError, match="problem_statement"):
        convert_row({"instance_id": "x", "metadata": {"instance_id": "x", "image": "i"}})


def test_convert_file_roundtrip(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    row = {
        "instance_id": "a__b-1",
        "n_resolved": 1,
        "metadata": {
            "instance_id": "a__b-1",
            "problem_statement": "p",
            "image": "img",
            "FAIL_TO_PASS": ["t"],
            "dataset_type": "swegym",
        },
    }
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    n = convert_file(src, dst)
    assert n == 1
    got = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
    assert set(got) >= {"prompt", "extra_info"}
