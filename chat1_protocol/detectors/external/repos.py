"""Third-party repositories used to test whether detector 3 generalizes.

Detector 3 was written against this project's own code, so its vocabularies
(VAL_NAMES, OOD_NAMES, EVAL_FNS) and its structural assumption -- a function
whose name contains a sweep hint, calling `train_model` inside a loop, then
calling an evaluation function whose first loader argument can be traced -- all
come from one codebase's conventions. The question this answers is whether it is
a detector or a project-specific grep.

Repositories are cloned READ-ONLY into the scratchpad and PARSED, never executed.
Commit SHAs are pinned so the result is reproducible.

Ground truth per file is established by a human reading the file, and the
`rationale` field records the reasoning. Where the authors themselves document
the behaviour (DomainBed's OracleSelectionMethod docstring), that is quoted
rather than inferred.

A note on scope: DomainBed is not clinical code. It is included because it is
the reference implementation for domain-generalization model selection and
contains an OOD-selecting method that its own authors label as an oracle, which
makes it the least ambiguous positive available anywhere. Its non-clinical
status is disclosed rather than glossed.
"""
import os

# Where the read-only clones live. Overridable, and it has to be: the default
# below was a session-specific scratch directory on the original author's
# machine, complete with username and a session UUID, which is not something a
# public repository should hardcode. It is also not durable -- the OS temp
# sweeper has emptied it twice during this project, which is why SCAN_CORPUS.md
# pins every commit SHA rather than relying on the clones surviving.
#
# Set PCL_SCRATCH_REPOS to any directory. `run_repos.py` reclones into it.
SCRATCH = os.environ.get(
    "PCL_SCRATCH_REPOS",
    os.path.join(os.path.expanduser("~"), ".cache", "pcl_scan_repos"))

TARGETS = [
    {
        "name": "DomainBed",
        "url": "https://github.com/facebookresearch/DomainBed",
        "commit": "b93c22a1cfc3b2428398272c1a116c8de1f4139e",
        "paper": "Gulrajani & Lopez-Paz, In Search of Lost Domain Generalization, ICLR 2021",
        "clinical": False,
        "files": [
            ("domainbed/model_selection.py", True,
             "Contains OracleSelectionMethod, whose own docstring states it "
             "'picks argmax(test_out_acc) across all hparams'. Selection is "
             "performed on the held-out TEST domain, which is precisely the "
             "confound detector 3 exists to find. The authors disclose it as an "
             "oracle, so ground truth here is documented, not inferred. The same "
             "file also contains IIDAccuracySelectionMethod, which selects on "
             "training-domain validation only, so a file-level verdict of "
             "CONTAMINATED is correct for the file as a whole."),
            ("domainbed/scripts/sweep.py", False,
             "Builds and launches job command lines across a hyperparameter "
             "grid. It never evaluates a model or compares scores, so no "
             "selection of any kind happens here; the choice of best run is "
             "made later in collect_results.py. A detector that flags this is "
             "reacting to the word 'sweep' rather than to selection."),
        ],
    },
    {
        "name": "mimic3-benchmarks",
        "url": "https://github.com/YerevaNN/mimic3-benchmarks",
        "commit": "ea0314c7cbd369f62e2237ace6f683740f867e3a",
        "paper": "Harutyunyan et al., Multitask learning and benchmarking with clinical time series data, Sci Data 2019",
        "clinical": True,
        "files": [
            ("mimic3models/in_hospital_mortality/logistic/main.py", False,
             "Fits a single LogisticRegression with C taken from a command-line "
             "argument. There is no loop over hyperparameters and therefore no "
             "selection at all. Both a validation and a test reader exist and "
             "both are scored, which is legitimate reporting -- the case "
             "detector 3 is specifically designed not to flag."),
            ("mimic3models/in_hospital_mortality/main.py", False,
             "Trains one Keras model per invocation with hyperparameters fixed "
             "from argparse, using a validation split for early stopping and "
             "checkpointing. Test data is read for final reporting only. No "
             "hyperparameter is chosen by comparing scores across candidates."),
        ],
    },
    {
        "name": "MIMIC_Extract",
        "url": "https://github.com/MLforHealth/MIMIC_Extract",
        "commit": "d8d2dea551283bea449b8495bfc1b5a41b90d837",
        "paper": "Wang et al., MIMIC-Extract, CHIL 2020",
        "clinical": True,
        "files": [],   # populated after inspection; see run_repos.py output
    },
    {
        "name": "HIRID-ICU-Benchmark",
        "url": "https://github.com/ratschlab/HIRID-ICU-Benchmark",
        "commit": "bee770094bf8389920bc09823895b87e09a563dd",
        "paper": "Yeche et al., HiRID-ICU-Benchmark, NeurIPS Datasets & Benchmarks 2021",
        "clinical": True,
        "files": [],   # populated after inspection; see run_repos.py output
    },
]


def resolved_files():
    """(repo, absolute path, expected_flag, rationale) for every declared file."""
    out = []
    for t in TARGETS:
        for rel, exp, why in t["files"]:
            out.append((t["name"], os.path.join(SCRATCH, t["name"], rel),
                        exp, why))
    return out
