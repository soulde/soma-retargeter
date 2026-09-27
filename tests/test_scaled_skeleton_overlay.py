from types import SimpleNamespace

import numpy as np
import warp as wp

import app.bvh_to_csv_converter as converter
from app.bvh_to_csv_converter import Viewer


class _ScaledInstance:
    def __init__(self):
        self.local_transforms = None

    def set_local_transforms(self, local_transforms):
        self.local_transforms = local_transforms


class _Scaler:
    def __init__(self, transforms):
        self.transforms = transforms
        self.calls = 0

    def compute_effectors_from_skeleton(self, source_instance, scale_animation):
        self.calls += 1
        assert scale_animation is True
        return self.transforms


class _Renderer:
    def __init__(self):
        self.draw_calls = []

    def draw(self, viewer, instance, renderer_id):
        self.draw_calls.append((viewer, instance, renderer_id))


def test_scaled_skeleton_overlay_updates_and_draws_only_when_enabled(monkeypatch):
    source_instance = SimpleNamespace(xform=object())
    robot_root = object()
    scaled_instance = _ScaledInstance()
    scaled_skeleton = object()
    global_transforms = np.array([[1.0, 2.0, 3.0]])
    local_transforms = np.array([[4.0, 5.0, 6.0]])
    scaler = _Scaler(global_transforms)
    renderer = _Renderer()
    fake_viewer = object()
    calls = []
    monkeypatch.setattr(
        converter,
        "pose_utils",
        SimpleNamespace(
            compute_local_pose=lambda skeleton, globals_, root:
                calls.append((skeleton, globals_, root)) or local_transforms
        ),
        raising=False,
    )

    viewer = object.__new__(Viewer)
    viewer.show_scaled_skeleton = False
    viewer.human_robot_scaler = scaler
    viewer.scaled_skeleton = scaled_skeleton
    viewer.scaled_skeleton_instances = [scaled_instance]
    viewer.scaled_skeleton_renderer = renderer
    viewer.skeleton_instances = [source_instance]
    viewer.robot_offsets = [robot_root]
    viewer.viewer = fake_viewer

    viewer._draw_scaled_skeleton_overlay(0)
    assert scaler.calls == 0
    assert renderer.draw_calls == []

    viewer.show_scaled_skeleton = True
    viewer._draw_scaled_skeleton_overlay(0)
    assert scaler.calls == 1
    assert len(calls) == 1
    assert calls[0][0] is scaled_skeleton
    assert calls[0][1] is global_transforms
    np.testing.assert_allclose(calls[0][2].p, (0.0, 0.0, 0.0))
    assert scaled_instance.local_transforms is local_transforms
    assert scaled_instance.xform is robot_root
    assert renderer.draw_calls == [(fake_viewer, scaled_instance, 1000)]


def test_source_preview_offset_does_not_move_scaled_overlay():
    source_instance = SimpleNamespace(xform=wp.transform_identity())
    seen = []

    class _Viewer:
        def begin_frame(self, time):
            pass

        def log_state(self, state):
            pass

        def end_frame(self):
            pass

    class _SourceRenderer:
        def draw(self, viewer, instance, index):
            seen.append(("source", float(instance.xform.p[0])))

    viewer = object.__new__(Viewer)
    viewer.viewer = _Viewer()
    viewer.time = 0.0
    viewer.state = object()
    viewer.animation_buffers = [object()]
    viewer.skeleton_instances = [source_instance]
    viewer.animation_offsets = [wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity())]
    viewer.skeleton_renderer = _SourceRenderer()
    viewer.show_skeleton = True
    viewer.show_skeleton_joint_axes = False
    viewer.show_skeleton_mesh = False
    viewer.show_gizmos = False
    viewer._draw_scaled_skeleton_overlay = lambda index: seen.append(
        ("scaled", float(source_instance.xform.p[0])))

    viewer.render()

    assert seen == [("source", 3.0), ("scaled", 0.0)]
    assert float(source_instance.xform.p[0]) == 0.0
