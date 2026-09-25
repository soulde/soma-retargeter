import ast
from pathlib import Path


def test_loading_bvh_recomputes_playback_duration():
    source = Path("app/bvh_to_csv_converter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    load_bvh = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_bvh_file"
    )

    calls = [
        node for node in ast.walk(load_bvh)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute_playback_total_time"
    ]
    assert calls, "loading a BVH must update the playback duration"
