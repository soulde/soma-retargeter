import ast
from pathlib import Path


def _function(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def test_retarget_mapping_supports_position_offsets():
    node = _function("soma_retargeter/pipelines/newton_pipeline.py", "_build_target_mapping")
    assert any(isinstance(n, ast.Constant) and n.value == "t_offset" for n in ast.walk(node))


def test_feet_stabilizer_supports_effector_offsets():
    node = _function("soma_retargeter/pipelines/feet_stabilizer.py", "_load_config")
    assert any(isinstance(n, ast.Constant) and n.value == "effector_offsets" for n in ast.walk(node))
