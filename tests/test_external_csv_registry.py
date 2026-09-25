from pathlib import Path

from soma_retargeter.assets.csv import get_csv_config_for_target
from soma_retargeter.pipelines import utils as pipeline_utils
from soma_retargeter.pipelines.utils import TargetRobot


class _FixtureCSVConfig:
    name = "fixture_csv"
    csv_header = ["Frame"]

    def to_anim_frame(self, csv_row):
        return csv_row

    def to_csv_row(self, frame_idx, anim_row):
        return [frame_idx]


def test_csv_config_can_be_provided_by_registered_target_package():
    previous = pipeline_utils._TARGET_REGISTRY.copy()
    previous_discovery_state = pipeline_utils._EXTERNAL_TARGETS_DISCOVERED
    try:
        pipeline_utils.register_target(
            TargetRobot(
                name="fixture_target",
                retargeter_config="fixture/config.json",
                get_config_base=lambda: Path("/tmp"),
                get_mjcf_path=lambda: Path("/tmp/fixture.xml"),
                csv_config_factory=_FixtureCSVConfig,
            )
        )
        config = get_csv_config_for_target("fixture_target")
        assert config.name == "fixture_csv"
    finally:
        pipeline_utils._TARGET_REGISTRY.clear()
        pipeline_utils._TARGET_REGISTRY.update(previous)
        pipeline_utils._EXTERNAL_TARGETS_DISCOVERED = previous_discovery_state
