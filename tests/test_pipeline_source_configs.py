import json

import pytest

from soma_retargeter.pipelines import utils


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_explicit_source_config_mapping_selects_lafan1(tmp_path):
    _write(tmp_path / "robot/lafan.json", {"kind": "lafan"})
    utils.register_target(utils.TargetRobot(
        name="mapped_robot",
        retargeter_config="robot/soma.json",
        get_config_base=lambda: tmp_path,
        get_mjcf_path=lambda: tmp_path / "robot.xml",
        retargeter_configs={"lafan1": "robot/lafan.json"},
    ))
    result = utils.get_retargeter_config(utils.SourceType.LAFAN1, "mapped_robot")
    assert result == {"kind": "lafan"}


def test_legacy_registration_still_resolves_soma_and_smplx(tmp_path):
    _write(tmp_path / "robot/soma_to_test.json", {"kind": "soma"})
    _write(tmp_path / "robot/smplx_to_test.json", {"kind": "smplx"})
    utils.register_target(utils.TargetRobot(
        name="legacy_robot",
        retargeter_config="robot/soma_to_test.json",
        get_config_base=lambda: tmp_path,
        get_mjcf_path=lambda: tmp_path / "robot.xml",
    ))
    assert utils.get_retargeter_config(utils.SourceType.SOMA, "legacy_robot") == {"kind": "soma"}
    assert utils.get_retargeter_config(utils.SourceType.SMPLX, "legacy_robot") == {"kind": "smplx"}


def test_missing_source_config_names_source_and_target(tmp_path):
    _write(tmp_path / "robot/soma.json", {"kind": "soma"})
    utils.register_target(utils.TargetRobot(
        name="soma_only_robot",
        retargeter_config="robot/soma.json",
        get_config_base=lambda: tmp_path,
        get_mjcf_path=lambda: tmp_path / "robot.xml",
    ))
    with pytest.raises(ValueError, match="lafan1.*soma_only_robot"):
        utils.get_retargeter_config(utils.SourceType.LAFAN1, "soma_only_robot")
