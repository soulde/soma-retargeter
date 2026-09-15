# LAFAN1 Source And Chocolate Retargeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a robot-independent LAFAN1 motion source and a production LAFAN1-to-Chocolate retargeting/export path.

**Architecture:** A LAFAN1 adapter validates and normalizes standard BVH into the existing `Skeleton` and `AnimationBuffer` contract. Target registration resolves an explicit source-to-config mapping while retaining legacy SOMA/SMPL-X behavior. Chocolate owns its LAFAN1 retargeter/scaler files, and the main repository owns generic loading, dispatch, tests, and batch export.

**Tech Stack:** Python 3.12, NumPy, Warp/Newton, SciPy, pytest, Bash, JSON configuration.

**Spec:** `docs/superpowers/specs/2026-09-15-lafan1-source-chocolate-design.md`

## Global Constraints

- Do not modify or overwrite `/home/jvwei/datasets/lafan1`, existing retargeted datasets, checkpoints, logs, or prior exports.
- Keep LAFAN1 source code robot-independent; Chocolate-specific files remain in `/home/jvwei/soma-chocolate`.
- Preserve compatibility with target plugins that provide only `retargeter_config`.
- Read FPS from BVH `Frame Time`; never label LAFAN1 as 120 Hz or split it into sampling phases.
- Export NPZ by direct source-rate-to-50-Hz resampling.
- Use a single agent, as explicitly requested by the user.
- Run only smoke tests before full generation; inspect resources before any full GPU job.

---

### Task 1: Explicit Source-To-Target Configuration Lookup

**Files:**
- Modify: `soma_retargeter/pipelines/utils.py`
- Create: `tests/test_pipeline_source_configs.py`

**Interfaces:**
- Consumes: existing `TargetRobot`, `register_target`, and `get_retargeter_config`.
- Produces: `TargetRobot.retargeter_configs: dict[str, str] | None` and explicit lookup for `soma`, `smplx`, and later `lafan1`.

- [ ] **Step 1: Write failing configuration lookup tests**

Cover an explicitly mapped source, legacy SOMA lookup, legacy SMPL-X filename fallback, and a missing source-target pair. Use temporary configuration roots and restore the private registry after each test. The missing mapping assertion must include both source and target names.

```python
def test_explicit_source_config_mapping_wins(tmp_path):
    write_json(tmp_path / "robot/lafan.json", {"kind": "lafan"})
    robot = TargetRobot(
        name="test_robot",
        retargeter_config="robot/soma.json",
        get_config_base=lambda: tmp_path,
        get_mjcf_path=lambda: tmp_path / "robot.xml",
        retargeter_configs={"lafan1": "robot/lafan.json"},
    )
    register_target(robot)
    assert get_retargeter_config(SourceType.LAFAN1, "test_robot")["kind"] == "lafan"
```

- [ ] **Step 2: Run tests and verify the expected failure**

Run: `pytest -q tests/test_pipeline_source_configs.py`

Expected: collection or assertion failure because `SourceType.LAFAN1` and `retargeter_configs` do not exist.

- [ ] **Step 3: Implement backward-compatible mapping**

Add an optional field after all required dataclass fields:

```python
retargeter_configs: dict[str, str] | None = None
```

Resolve an explicit mapping first. Preserve `retargeter_config` for SOMA and the current `soma_to_` to `smplx_to_` compatibility fallback. Raise a `ValueError` naming the unsupported source-target pair rather than selecting another source.

- [ ] **Step 4: Run focused and existing tests**

Run:

```bash
pytest -q tests/test_pipeline_source_configs.py tests/test_smplx_loader.py tests/test_beyondmimic_npz.py
```

Expected: all pass.

- [ ] **Step 5: Commit the generic registration change**

```bash
git add soma_retargeter/pipelines/utils.py tests/test_pipeline_source_configs.py
git commit -m "feat: support source-specific retarget configs"
```

---

### Task 2: LAFAN1 Loader Contract

**Files:**
- Create: `soma_retargeter/assets/lafan1.py`
- Modify: `soma_retargeter/assets/__init__.py`
- Create: `tests/test_lafan1_loader.py`

**Interfaces:**
- Consumes: `soma_retargeter.assets.bvh.load_bvh(path, input_skeleton=None)`.
- Produces: `load_lafan1_bvh(path: str | Path, input_skeleton: Skeleton | None = None) -> tuple[Skeleton, AnimationBuffer]` and `validate_lafan1_skeleton(skeleton: Skeleton) -> None`.

- [ ] **Step 1: Write failing loader tests**

Use `/home/jvwei/datasets/lafan1/walk1_subject1.bvh` behind a skip marker when unavailable. Assert:

```python
assert animation.sample_rate == pytest.approx(30.0, abs=1e-3)
assert skeleton.joint_names == list(STANDARD_LAFAN1_JOINT_NAMES)
assert skeleton.num_joints == 22
assert np.asarray(skeleton.up_axis).tolist() == [0.0, 0.0, 1.0]
assert np.asarray(skeleton.forward_axis).tolist() == [0.0, -1.0, 0.0]
```

Add synthetic skeleton tests that reject a missing `LeftToe`, duplicate names,
and a changed parent. Add a fixture with a different positive `Frame Time` and
assert the loader preserves its rate.

- [ ] **Step 2: Run tests and verify import failure**

Run: `pytest -q tests/test_lafan1_loader.py`

Expected: FAIL because `soma_retargeter.assets.lafan1` does not exist.

- [ ] **Step 3: Implement the thin LAFAN1 adapter**

Define the literal 22-name order:

```python
STANDARD_LAFAN1_JOINT_NAMES = (
    "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
)
```

Call the shared BVH loader, validate names/topology/sample rate and finite local
transforms, and set the documented axes. Do not resample or introduce
Chocolate-specific aliases in this module.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_lafan1_loader.py tests/test_smplx_loader.py`

Expected: all pass.

- [ ] **Step 5: Commit the loader**

```bash
git add soma_retargeter/assets/lafan1.py soma_retargeter/assets/__init__.py tests/test_lafan1_loader.py
git commit -m "feat: add LAFAN1 motion source loader"
```

---

### Task 3: Batch And Viewer Source Dispatch

**Files:**
- Modify: `app/bvh_to_csv_converter.py`
- Modify: `soma_retargeter/pipelines/utils.py`
- Create: `tests/test_lafan1_dispatch.py`

**Interfaces:**
- Consumes: `load_lafan1_bvh`, `SourceType.LAFAN1`, and the source-specific config lookup.
- Produces: `retarget_source: "lafan1"` support in batch and viewer loading.

- [ ] **Step 1: Write failing dispatch tests**

Extract or exercise a small source descriptor API so tests assert observable
dispatch rather than source text:

```python
descriptor = motion_source_descriptor("lafan1")
assert descriptor.extension == ".bvh"
assert descriptor.load is load_lafan1_bvh
assert descriptor.root_transform_is_identity is True
```

Also assert `get_source_model_mesh(SourceType.LAFAN1, skeleton) is None`.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_lafan1_dispatch.py`

Expected: FAIL because no LAFAN1 descriptor exists.

- [ ] **Step 3: Implement source dispatch**

Add `LAFAN1` to `SourceType`, string maps, and viewer options. Introduce a
focused descriptor or equivalent centralized dispatch that provides extension,
loader, reference-skeleton creation, and root transform policy. Replace the
duplicated SOMA/SMPL-X branches in batch loading with that descriptor. LAFAN1
uses the identity transform because the adapter already emits pipeline axes.

- [ ] **Step 4: Run dispatch and repository tests**

Run: `pytest -q tests/test_lafan1_dispatch.py tests`

Expected: all pass.

- [ ] **Step 5: Commit source dispatch**

```bash
git add app/bvh_to_csv_converter.py soma_retargeter/pipelines/utils.py tests/test_lafan1_dispatch.py
git commit -m "feat: dispatch LAFAN1 batch motions"
```

---

### Task 4: Chocolate LAFAN1 Configuration

**Files:**
- Modify: `/home/jvwei/soma-chocolate/soma_chocolate/__init__.py`
- Create: `/home/jvwei/soma-chocolate/soma_chocolate/configs/chocolate/lafan1_to_chocolate_retargeter_config.json`
- Create: `/home/jvwei/soma-chocolate/soma_chocolate/configs/chocolate/lafan1_to_scaler_config.json`
- Create: `/home/jvwei/soma-chocolate/tests/test_lafan1_config.py`

**Interfaces:**
- Consumes: explicit `TargetRobot.retargeter_configs`, standard LAFAN1 joint names, existing Chocolate MJCF and post-processing files.
- Produces: a registered `lafan1 -> chocolate` configuration pair.

- [ ] **Step 1: Write failing Chocolate config contract tests**

Load the MJCF with Newton and both JSON files. Assert every scaler joint is a
standard LAFAN1 joint, every `ik_map` key is a scaler joint, every target body
exists in the Chocolate model, `feet_joint_names == ["LeftToe", "RightToe"]`,
and registration maps `lafan1` to the new retargeter file.

- [ ] **Step 2: Run tests and verify missing-file failure**

Run from `/home/jvwei/soma-chocolate`:

```bash
source /home/jvwei/soma-retargeter/.venv/bin/activate
PYTHONPATH=/home/jvwei/soma-retargeter:. pytest -q tests/test_lafan1_config.py
```

Expected: FAIL because the LAFAN1 Chocolate files are absent.

- [ ] **Step 3: Create the scaler configuration**

Use the exact LAFAN1 hierarchy rooted at `Hips`. Include root, torso, bilateral
hip/knee/foot/toe, shoulder/elbow/hand targets. Derive scales from actual LAFAN1
rest offsets and the tuned Chocolate proportions. Derive quaternion offsets by
aligning a representative LAFAN1 pose to the existing Chocolate target frames;
normalize every quaternion and preserve bilateral symmetry where the source
axes permit it.

- [ ] **Step 4: Create the retargeter configuration and registration**

Copy the robot-side iteration, joint-limit, smoothing, post-processing, and
feet-stabilizer settings from `smplx_to_chocolate_retargeter_config.json`.
Replace only source-dependent initialization, scaler, `ik_map`, and foot names.
Register all three Chocolate sources explicitly:

```python
retargeter_configs={
    "soma": "chocolate/soma_to_chocolate_retargeter_config.json",
    "smplx": "chocolate/smplx_to_chocolate_retargeter_config.json",
    "lafan1": "chocolate/lafan1_to_chocolate_retargeter_config.json",
}
```

- [ ] **Step 5: Run Chocolate config tests**

Run the focused command from Step 2 and the existing soma-retargeter tests.
Expected: all pass.

- [ ] **Step 6: Commit the Chocolate package change separately**

```bash
git -C /home/jvwei/soma-chocolate add soma_chocolate/__init__.py soma_chocolate/configs/chocolate/lafan1_to_chocolate_retargeter_config.json soma_chocolate/configs/chocolate/lafan1_to_scaler_config.json tests/test_lafan1_config.py
git -C /home/jvwei/soma-chocolate commit -m "feat: configure LAFAN1 retargeting"
```

---

### Task 5: Representative End-To-End Smoke Validation

**Files:**
- Create: `tests/test_lafan1_chocolate_integration.py`
- Create: `configs/lafan1_chocolate_smoke.json`

**Interfaces:**
- Consumes: `load_lafan1_bvh`, `NewtonPipeline(source_type="lafan1", robot_type="chocolate")`, and `csv_to_npz.py`.
- Produces: evidence that walk, run, and dance reach valid Chocolate CSV/NPZ outputs.

- [ ] **Step 1: Write the integration assertions before tuning**

For local fixtures `walk1_subject1.bvh`, `run1_subject2.bvh`, and
`dance1_subject1.bvh`, retarget a bounded prefix into a temporary directory.
Assert finite qpos, 23 joint coordinates, source-rate CSV duration, finite
50 Hz NPZ arrays, model joint limits, plausible root height, and left/right
foot values. Mark only missing local data as skip; numerical failures must fail.

- [ ] **Step 2: Run a short walk smoke test and capture the initial failure**

Run:

```bash
CUDA_VISIBLE_DEVICES= WARP_CACHE_PATH=/tmp/soma-lafan1-warp-cache \
pytest -q tests/test_lafan1_chocolate_integration.py -k walk
```

Expected: initial numerical or contract failure identifies the first calibration issue.

- [ ] **Step 3: Tune one calibration variable at a time**

Adjust only LAFAN1 Chocolate scaler/rotation/ground offsets supported by the
failed assertion. Re-run the walk case after every individual adjustment. Do
not weaken joint-limit, finite-value, duration, or left-right assertions.

- [ ] **Step 4: Run walk, run, and dance smoke tests**

Run the full integration file with CPU first. If a bounded GPU smoke test is
needed, inspect `nvidia-smi`, `free -h`, and `pueue status`, then use a distinct
temporary output directory without interrupting existing work.

- [ ] **Step 5: Visually inspect representative outputs**

Replay the three candidate outputs and verify forward direction, no left-right
swap, plausible knee motion, forward feet, and no sustained ground
penetration. Record observed pass/fail facts; do not accept configuration based
only on file generation.

- [ ] **Step 6: Commit integration coverage and finalized configs**

Commit the integration test/config in `soma-retargeter`; commit any final
calibration delta in `soma-chocolate` separately.

---

### Task 6: Resumable 77-Clip Export And Manifest Validation

**Files:**
- Create: `scripts/export_lafan1_chocolate_npz50.sh`
- Create: `tests/test_lafan1_export_workflow.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: batch retarget config, Chocolate CSV exporter, and direct `csv_to_npz.py --input-fps <source-fps> --output-fps 50`.
- Produces: separate LAFAN1 Chocolate CSV and NPZ directories with one output per source file.

- [ ] **Step 1: Write a failing workflow test**

Run the wrapper against a temporary one-file fixture with smoke mode. Assert
one CSV and one NPZ with the same stem, `fps=50`, correct duration-derived frame
count, no phase suffix, refusal to overwrite by default, and explicit resume
skipping an existing verified output.

- [ ] **Step 2: Run the workflow test and verify missing-script failure**

Run: `pytest -q tests/test_lafan1_export_workflow.py`

Expected: FAIL because the wrapper is absent.

- [ ] **Step 3: Implement the export wrapper**

Use explicit defaults:

```text
input: /home/jvwei/datasets/lafan1
csv:   /home/jvwei/datasets/lafan1_retargeted/chocolate_csv30
npz:   /home/jvwei/datasets/lafan1_retargeted/chocolate_npz50
```

The wrapper activates `/home/jvwei/soma-retargeter/.venv`, records all paths,
uses the LAFAN1 batch config, reads each CSV's associated source FPS, converts
directly to 50 Hz, refuses overwrite, and supports `RESUME=1` only after basic
output validation.

- [ ] **Step 4: Run workflow and full test suites**

Run soma-retargeter tests and the Chocolate package tests. Expected: all pass.

- [ ] **Step 5: Inspect resources and run one-file smoke export**

Run `nvidia-smi`, `free -h`, and `pueue status`. Export only
`walk1_subject1.bvh` first and validate its CSV and NPZ before considering the
full dataset.

- [ ] **Step 6: Run the full export only when resources permit**

Use the project wrapper without overwriting existing outputs. Do not overlap a
heavy GPU retarget job with an occupied GPU. If the full job is deferred,
report the exact validated command and blocker instead of claiming generation.

- [ ] **Step 7: Validate the final manifest**

Check 77 source BVHs map one-to-one to 77 CSVs and 77 NPZs. For every NPZ,
assert all BeyondMimic fields exist, arrays are finite, `fps=50`, and frame
count matches source duration after resampling.

- [ ] **Step 8: Document usage and commit**

Document the source contract, batch command, paths, direct 30-to-50 behavior,
resume semantics, and validation command. Commit only code/docs/tests, never
generated data.

---

### Task 7: Final Verification

**Files:**
- Verify all files changed by Tasks 1-6.

**Interfaces:**
- Consumes: completed generic source framework, Chocolate configuration, and export workflow.
- Produces: final evidence and a clean scoped handoff.

- [ ] **Step 1: Run fresh full tests in both repositories**

Run:

```bash
cd /home/jvwei/soma-retargeter
WARP_CACHE_PATH=/tmp/soma-lafan1-warp-cache pytest -q
cd /home/jvwei/soma-chocolate
PYTHONPATH=/home/jvwei/soma-retargeter:. WARP_CACHE_PATH=/tmp/soma-lafan1-warp-cache pytest -q
```

- [ ] **Step 2: Check diffs and repository state**

Run `git diff --check` and `git status --short` in both repositories. Preserve
and identify unrelated pre-existing changes.

- [ ] **Step 3: Report evidence**

Report test counts, representative smoke results, generated artifact counts if
the full export ran, exact output paths, commits in both repositories, and any
resource blocker. Do not claim visual quality without completing playback.
