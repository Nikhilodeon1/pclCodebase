"""Committed logs must not carry retracted phrasing or retracted verdicts.

Both failures this guards against actually happened. `detectors/logs/check5.out`
sat in the repository asserting "DETECTOR VALIDATED" and a composition share of
0.49 long after the alphabetical-prefix sampling bug was fixed and the 0.49
figure was retracted; the result of record by then was a FALSE NEGATIVE. A
reader listing the logs would have found a machine-generated artifact
contradicting the paper's own headline limitation.

Scanned files are limited to `detectors/logs/*.out` on purpose. Prose files
discuss the retracted quantity legitimately -- PREREGISTRATION_OUTCOME.md
explains why it is not a share -- so scanning them would flag the very
documentation that records the retraction. A log is different: the phrase can
only appear there because the code emitted it, which means the log predates the
fix.
"""
import glob
import io
import os
import unittest

LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "logs")

# (filename glob the rule applies to, substring, why it must never appear)
#
# "DETECTOR VALIDATED" is scoped to check5 logs deliberately. It is the generic
# pass verdict every check prints, and checks 1 and 2 legitimately print it --
# an unscoped ban flagged five healthy logs when this test was first written.
# Only for detector 5 does it contradict the result of record.
BANNED = [
    ("*.out", "share of gap explained",
     "composition_gap_ratio is not a share and can exceed 1.0 (E4 = 1.414); "
     "this wording was retracted"),
    ("*.out", "share of the gap explained",
     "same retraction, alternate wording"),
    ("check5*.out", "DETECTOR VALIDATED",
     "detector 5's positive case is a documented FALSE NEGATIVE; a check5 log "
     "claiming validation predates the sampling fix"),
]


class TestLogPhrasing(unittest.TestCase):
    def test_no_retracted_phrasing_in_logs(self):
        self.assertTrue(sorted(glob.glob(os.path.join(LOGS, "*.out"))),
                        "no logs found -- the guard would pass vacuously")
        offenders = []
        for pattern, phrase, why in BANNED:
            files = sorted(glob.glob(os.path.join(LOGS, pattern)))
            self.assertTrue(files, f"no log matches {pattern!r}: this rule "
                                   "would pass vacuously")
            for path in files:
                text = io.open(path, encoding="utf-8", errors="replace").read()
                if phrase in text:
                    offenders.append(f"{os.path.basename(path)}: "
                                     f"{phrase!r} — {why}")
        self.assertEqual(offenders, [],
                         "stale committed logs:\n  " + "\n  ".join(offenders))


if __name__ == "__main__":
    unittest.main()
