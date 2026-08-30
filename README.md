# PCL (Physiology-Constrained Learning) — repo index

Physiology-Constrained Learning: a representation learning framework that
enforces hemodynamic and biochemical laws (Mean Arterial Pressure,
Henderson-Hasselbalch, Severinghaus) during masked transformer pretraining, for
cross-hospital generalization on clinical time-series. The original headline
result did not survive clean labels — see `chat2_papers/` for the corrected
story.

This is not a single runnable pipeline anymore. It's a shared workspace split
across four active work-streams plus one frozen archive. **Start with
[PROJECTS.md](PROJECTS.md)** — one table, every active paper, nothing else.
That file is the fix for exactly the "wait, is this a third project?"
confusion that's happened before here — if you add or retire a project,
update it there, not just in your head.

## Where things are

| Folder | What | Start here |
|---|---|---|
| `chat1_protocol/` | Confound-detection methods paper — five detectors, each validated against known ground truth. Code + data + tests. | [chat1_protocol/README.md](chat1_protocol/README.md) |
| `chat2_papers/` | The URTC paper (submitted) and the withdrawn NeurIPS draft. No datasets, no compute — self-contained manuscript build. | [chat2_papers/README.md](chat2_papers/README.md) |
| `pcl-legacy2/` | Cross-task generalization of URTC's selection-criterion finding. GPU/RunPod project, blocked on the pod. | [pcl-legacy2/README.md](pcl-legacy2/README.md) |
| `deadpcl/` | ML4H Findings negative-results writeup of the original PCL investigation. | [deadpcl/README.md](deadpcl/README.md) |
| `_archive/` | Frozen pre-split pipeline (`run_paper_experiments.py`, old `src/`, old results). Superseded — kept for history only. | [_archive/README.md](_archive/README.md) |
| `pod_monitor.py` | Shared GPU-idle/active watchdog — import into any GPU-pod script to get a loud switch-pods alert in either direction. See its docstring. | — |
| `data/` | Shared 455MB dataset (PhysioNet full, eICU-demo, MIMIC-IV demo). | — |
| `venv/` | Shared Python venv. Install with root `requirements.txt`. | — |

Each project folder owns its own files; none of them modify another's.

## Hazard: the `data/` junctions

`chat1_protocol/data` (and any other `*/data` you find) is a **Windows
directory junction** into the one real `data/` at repo root — not a copy.
**Never `rm -rf` a junction path** — on Windows that can follow the link and
delete the real 455MB dataset. To remove a junction, use:

```bash
cmd //c rmdir "chat1_protocol\data"
```

## Setup

```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Then follow whichever sub-folder's README matches what you're working on.
