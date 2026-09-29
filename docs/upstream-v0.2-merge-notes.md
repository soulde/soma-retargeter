# NVIDIA v0.2.0 merge — 2026-09-29

Upstream: `1733b820f3cdf6f74bbc81a10bda3201b38c7bcf` (`v0.2.0`).
Local pre-merge HEAD: `f23f8f9`.

Upstream rewrote its earlier history. The old base `9dd02be` and the new-history
commit `b3ef270` have exactly the same tree,
`fc54a553f82366836f3b2e8128e2f5896071b9ce`. A temporary, private replacement-ref
namespace made that equivalent base available to Git's normal three-way merge.
The temporary ref was removed after conflict resolution. No existing commit was
rewritten; the merge retains the original local and current upstream parents.

Conflicts in the viewer entry point, pipeline utilities, scaler, README, assets
package and LFS attributes use the upstream version. The removed legacy
`NewtonPipeline`, `FeetStabilizer`, and CSV module are not reintroduced into the
new pipeline. Local files without conflicts remain, including source adapters,
export scripts and their tests. The two local G1 SMPL-X configuration files follow
Git's detected directory move; their contents have not been converted to v0.2.

## Preserved local work

All 13 modified tracked files and 26 untracked files from before the merge are
restored in `.worktrees/pre-upstream-20260929` on
`backup/pre-upstream-20260929`. The tracked patch and every untracked regular
file were compared against the pre-merge backup. The stash is retained as an
additional recovery copy. Nothing in the external dataset directories or the
separate `soma-chocolate` repository was changed for this merge.

To use the old code with the existing environment, explicitly select that source
tree; the existing environment's editable installation otherwise points here:

```bash
cd /home/jvwei/soma-retargeter/.worktrees/pre-upstream-20260929
PYTHONPATH="$PWD" /home/jvwei/soma-retargeter/.venv/bin/python app/bvh_to_csv_converter.py --config assets/lafan1_chocolate_smoke.json --viewer gl
```

## Compatibility boundary

The converter exposes SOMA, SMPL-X, and LAFAN1, dispatches each file to its
source loader, and filters robot choices by the selected source configuration.
SMPL-X and LAFAN1 retain their native 22-joint skeletons. The source adapters
normalize coordinate axes, then the v0.2 pipeline maps each skeleton directly
to robot effectors. The scaled-skeleton overlay uses the upstream scaler's
`create_scaled_skeleton()` and effector computation.

The registry adapts installed `soma_retargeter.targets` entry points and keeps
the package's CSV config factory available to the v0.2 GUI. Version-aware
packages can register v0.2 configs while older clients keep using their
original config files. Chocolate's v0.2-format configs are generated beside
the old files under `configs/chocolate/v02`; the generator preserves calibrated
effector offsets and IK mappings and uses the upstream uniform scaler. Legacy
per-joint scale multipliers are not carried forward because v0.2 has no such
runtime behavior.

The local Unitree G1 SMPL-X assets are registered using v0.2-format config
files. Retargeting smoke checks used the isolated v0.2 environment on GPU: a
12-frame LAFAN1 clip retargeted to Chocolate and a 5-frame native SMPL-X clip
retargeted to G1. Both produced finite output; the latter retained the SMPL-X
joint names throughout loading and retargeting.

The original `.venv` was not upgraded. Upstream validation uses the isolated
environment `/tmp/soma-upstream-v02-env`, synchronized from the upstream lockfile
(Newton 1.3.0 and Warp 1.14.0). Recreate it with:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/soma-upstream-v02-env uv sync --locked
```

Validation logs, smoke outputs, stash ID, pre-merge patch and untracked archive
are under `/tmp/soma-upstream-merge-20260929`.

The original `.venv` was not upgraded. Upstream validation uses the isolated
environment `/tmp/soma-upstream-v02-env`, synchronized from the upstream lockfile
(Newton 1.3.0 and Warp 1.14.0). Recreate it with:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/soma-upstream-v02-env uv sync --locked
```

Validation logs, the 12-frame upstream smoke input/config/output, stash ID,
pre-merge patch and untracked archive are under
`/tmp/soma-upstream-merge-20260929`.
