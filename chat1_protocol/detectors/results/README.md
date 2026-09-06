# What is in this directory

Aggregate results only — confusion matrices, kappa values, confidence intervals,
t-statistics, flag rates, per-repository file counts. No per-stay or per-patient
data. `.gitignore` excludes `*.npz`, `*.pkl` and `*.pt` here for that reason; see
`../TODO.md` section 0 to regenerate those with your own credentials.

`run_all.py` reads `check{1,2,5}.json` for its cached rows and re-runs checks 3
and 4 live. It needs nothing else in this directory and no clinical data.

## Scale — read this before quoting a number

Some files are demo-scale and some are full-scale. **They are not
interchangeable, and the headline figures in `../../README.md` are the
full-scale ones.**

    file                     scale   notes
    check1.json              demo    PhysioNet; the paper's full-scale row is in README
    check2.json              demo    PhysioNet Site A -> Site B
    check3.json              n/a     static analysis, scale-free
    check4.json              n/a     static analysis, scale-free
    check5.json              demo    PhysioNet, 1200 stays/arm, seed 0
    bootstrap_kappa.json     DEMO    n=117, kappa 0.651, P(flag)=0.216
    external1.json           demo    MIMIC-IV demo + eICU demo
    external2.json           demo    MIMIC-IV demo (117 stays) -> eICU demo
    external5.json           demo    eICU demo with controlled HCO3 ablation
    baselines.json           demo    the three standard checks, for comparison
    check2_fill_audit.json   demo    missingness confound audit for detector 2
    check5_sampling.json     n/a     sampling-sensitivity study, n=4000
    external3_repos.json     n/a     detector 3 on the original 4-file corpus
    external4_lineage.json   n/a     detector 4 lineage traces, own + external code
    scan_repos.json          n/a     detector 3 corpus scan, 735 files / 18 repos

`bootstrap_kappa.json` is the one most likely to be misread. It is the DEMO
bootstrap: kappa 0.651, CI [0.528, 0.776], P(flag) = 0.216 at n=117. The paper
reports the FULL-scale figures — kappa 0.602, audit-500 CI [0.539, 0.663],
**P(flag) = 0.484** — which are transcribed from run output, because the
full-scale logs were lost (see "Archival gap" in the top-level README). The
demo-scale file is kept because it is a real archived artifact; it is not the
result of record.

The full-scale runs that ARE archived are logs, not JSON:
`../logs/full2.out` (detector 2) and `../logs/full5.out` (detector 5).
