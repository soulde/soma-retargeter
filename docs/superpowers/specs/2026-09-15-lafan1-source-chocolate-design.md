# LAFAN1 Source And Chocolate Retargeting Design

## Goal

Add LAFAN1 as a first-class, robot-independent motion source in
`soma-retargeter`. The first production source-target combination is LAFAN1 to
Chocolate. Future robots must be able to add LAFAN1 support by registering
configuration files without changing the LAFAN1 loader or batch pipeline.

The source dataset is the 77 BVH files under `/home/jvwei/datasets/lafan1`.
Generated CSV and NPZ artifacts are local data and are not committed.

## Scope

This change includes:

- a `lafan1` source type;
- LAFAN1-specific loading and validation on top of the shared animation types;
- explicit source-to-target retargeter configuration registration;
- backward compatibility for existing target registrations;
- LAFAN1-to-Chocolate retargeter and scaler configurations;
- batch retargeting from LAFAN1 BVH to Chocolate CSV;
- direct resampling from the retargeted source rate to 50 Hz AMP NPZ;
- automated contract tests and representative motion smoke tests;
- a resumable, non-overwriting batch export command for all 77 files.

This change does not add LAFAN1 configurations for G1 or DR02 and does not
modify existing SOMA, SMPL-X, KIT, or four-phase datasets.

## Architecture

The data flow is:

```text
LAFAN1 BVH
  -> LAFAN1 loader and convention validation
  -> Skeleton + AnimationBuffer
  -> NewtonPipeline(source=lafan1, target=chocolate)
  -> source-rate Chocolate CSV
  -> direct source-rate-to-50-Hz resampling
  -> BeyondMimic-compatible AMP NPZ
```

The LAFAN1 layer owns only source semantics. It must not import Chocolate or
contain robot-specific mappings. The Newton pipeline remains shared across
all source and target combinations.

## LAFAN1 Source Contract

The loader consumes standard LAFAN1 BVH files and returns the existing
`(Skeleton, AnimationBuffer)` contract. It must:

- preserve the BVH `Frame Time` as the animation sample rate;
- convert BVH translations from centimeters to meters exactly once;
- emit the pipeline convention of Z-up and forward negative Y;
- retain the standard 22-joint LAFAN1 names and parent topology;
- validate that clips used in one batch have compatible names and topology;
- reject missing, duplicated, or non-finite joint data with a path-specific
  error;
- expose stable left and right foot semantics using `LeftFoot`, `LeftToe`,
  `RightFoot`, and `RightToe` without changing the temporal samples.

The loader may reuse the generic BVH parser, but the LAFAN1 contract and
coordinate checks live in a separate source module so generic BVH behavior is
not silently changed.

## Source And Target Configuration Registration

`TargetRobot` will support an explicit mapping from source name to retargeter
configuration path. Conceptually:

```python
retargeter_configs = {
    "soma": "chocolate/soma_to_chocolate_retargeter_config.json",
    "smplx": "chocolate/smplx_to_chocolate_retargeter_config.json",
    "lafan1": "chocolate/lafan1_to_chocolate_retargeter_config.json",
}
```

Existing plugins that register only `retargeter_config` remain valid. That
legacy field supplies the SOMA configuration, while the existing SMPL-X
filename convention remains available as a compatibility fallback. New
registrations should use the explicit mapping.

Configuration lookup must report an unsupported source-target pair directly;
it must not silently select another source's configuration. Adding LAFAN1 for
a future robot requires only its model registration and LAFAN1 retargeter and
scaler configuration files.

## LAFAN1-To-Chocolate Configuration

The Chocolate configuration inherits robot-side decisions from the tuned
SMPL-X-to-Chocolate pipeline:

- robot model and root link;
- IK objective weights;
- joint-limit objective;
- temporal smoothing;
- foot stabilization;
- post-processing behavior.

Human-side targets use LAFAN1 names. At minimum the mapping covers the root,
torso, shoulders, elbows, hands, hips, knees, feet, and toes. The scaler
configuration uses the LAFAN1 hierarchy and contains explicit per-joint scales,
parents, translation offsets, and quaternion offsets.

The initial offsets are derived from the current Chocolate SMPL-X
configuration and the GMR LAFAN1 references, then validated against actual
Chocolate poses. GMR values are references, not copied blindly, because the
Newton and GMR solvers use different target and offset conventions.

Representative calibration clips are:

- `walk1_subject1.bvh` for gait, foot contact, and heading;
- `run1_subject2.bvh` for faster transitions and joint limits;
- `dance1_subject1.bvh` for upper-body coverage and orientation range.

Acceptance requires correct forward direction, no left-right swap, plausible
knee flexion, forward-facing feet, finite output, and no sustained ground
penetration. Candidate configuration output is kept separate from production
configuration until these checks pass.

## Batch Processing And Output

Batch mode recognizes `retarget_source: "lafan1"` and discovers `.bvh` files
recursively. It preserves relative paths and source filenames in its CSV
output. The retargeted CSV retains each clip's actual source frame rate.

The NPZ export resamples directly from that rate to 50 Hz. LAFAN1 data is not
labelled as 120 Hz, split into phases, or passed through an intermediate 30 Hz
decimation step.

The production batch wrapper uses distinct output directories, refuses to
overwrite by default, and supports an explicit resume mode that skips verified
existing outputs. It records the input root, output roots, robot, source type,
input rate, output rate, and configuration in its command or log.

## Error Handling

The pipeline stops before writing a clip when:

- the BVH skeleton violates the LAFAN1 contract;
- frame time is missing, non-finite, or non-positive;
- the selected target has no LAFAN1 configuration;
- configuration human joints are absent from the source skeleton;
- configuration robot bodies are absent from the target model;
- retargeted values are non-finite;
- an output path already exists without resume mode.

Batch errors include the source path. Successfully written artifacts are not
deleted when a later clip fails, allowing diagnosis and explicit resume.

## Verification

Unit and contract tests cover:

- `lafan1` source string and enum conversion;
- standard 22-joint names and topology;
- centimeter-to-meter conversion;
- Z-up and negative-Y-forward convention;
- sample-rate extraction from BVH `Frame Time`;
- explicit source-target configuration lookup;
- backward compatibility for existing target registrations;
- all LAFAN1 human targets referenced by the Chocolate configuration;
- all Chocolate bodies and feet referenced by the configuration;
- direct source-rate-to-50-Hz frame counts and NPZ metadata;
- refusal to overwrite and successful resume behavior.

Integration smoke tests retarget walk, run, and dance samples and check finite
qpos, target dimensions, joint limits, duration preservation, foot-height
statistics, and left-right consistency. Visual playback is used for calibration
acceptance, not as a substitute for automated checks.

After smoke tests pass, all 77 files are exported to new directories. Final
validation requires a one-to-one source/output manifest, `fps=50` in every NPZ,
complete BeyondMimic fields, finite arrays, and frame counts consistent with
each source duration.

## Data And Resource Safety

Source BVH files, existing datasets, checkpoints, and previous exports remain
unchanged. Smoke tests write only to temporary or explicitly named candidate
directories. Before full GPU retargeting, inspect GPU, system memory, and the
existing task queue; do not overlap a heavy retarget job with an occupied GPU.
