import inspect

from soma_retargeter.pipelines.feet_stabilizer import FeetStabilizer


def test_feet_stabilizer_resolves_registered_targets_without_name_allowlist():
    source = inspect.getsource(FeetStabilizer.__init__)
    assert "get_robot_mjcf_path(self.robot_type)" in source
    assert "SUPPORTED_ROBOT_TYPES" not in source
