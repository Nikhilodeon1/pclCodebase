"""
Build the URTC submission from the official IEEE letter template.

Strategy: edit the template in place rather than generate a document from
scratch. The IEEE template encodes its layout in a chain of seven section
breaks (full-width title, a four-column author grid, then the two-column body);
rebuilding that from scratch risks silently violating the format. So we keep the
template's header region (title / author block / abstract / keywords) byte-for-
byte, substitute only its text nodes, then replace the body with our content
using the template's own style IDs (Heading1, BodyText, tablehead, references, ...).
"""
import os
import re
import shutil
import subprocess
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "conference-template-letter.docx")
WORK = os.path.join(HERE, "_urtc_build")
OUT = os.path.join(HERE, "URTC_paper.docx")


def esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def runs(text, bold=False, italic=False):
    rpr = ""
    if bold:
        rpr += "<w:b/>"
    if italic:
        rpr += "<w:i/>"
    rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def para(text="", style="BodyText", bold=False, italic=False, extra_runs=None,
         keep=False):
    body = extra_runs if extra_runs is not None else (runs(text, bold, italic) if text else "")
    k = "<w:keepLines/>" if keep else ""
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{k}</w:pPr>{body}</w:p>'


def cell(text, w, style="tablecopy", bold=False, align=None):
    jc = f'<w:jc w:val="{align}"/>' if align else ""
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>'
            f'<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{jc}</w:pPr>'
            f'{runs(text, bold)}</w:p></w:tc>')


def picture(rid, w_emu, h_emu, name="Figure"):
    """Inline image run. Sized in EMU (914400 per inch)."""
    return (
        # NOTE: the template's BodyText style fixes the line height (w:line=11.40pt),
        # which clips an inline image taller than that line box. The image
        # paragraph therefore uses no body style and an automatic line rule.
        '<w:p><w:pPr><w:keepNext/><w:keepLines/>'
        '<w:spacing w:before="120" w:after="80" w:line="240" w:lineRule="auto"/>'
        '<w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{w_emu}" cy="{h_emu}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{abs(hash(rid))%9000+1}" name="{name}"/><wp:cNvGraphicFramePr/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{abs(hash(rid))%9000+1}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{w_emu}" cy="{h_emu}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def table(rows, widths, header_style="tablecolhead"):
    """Simple bordered table; first row is the header."""
    borders = ("<w:tblBorders>" + "".join(
        f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        for s in ("top", "left", "bottom", "right", "insideH", "insideV")) +
        "</w:tblBorders>")
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/>'
           f'<w:jc w:val="center"/>{borders}</w:tblPr>'
           f'<w:tblGrid>{grid}</w:tblGrid>']
    for ri, row in enumerate(rows):
        hdr = '<w:trPr><w:tblHeader/></w:trPr>' if ri == 0 else ""
        st = header_style if ri == 0 else "tablecopy"
        cells = "".join(
            cell(c, widths[ci], st, bold=(ri == 0),
                 align=("center" if ci > 0 or ri == 0 else None))
            for ci, c in enumerate(row))
        out.append(f"<w:tr>{hdr}{cells}</w:tr>")
    out.append("</w:tbl>")
    # Word collapses the gap between a table and the following paragraph;
    # an explicit small empty paragraph restores it.
    out.append('<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="120" w:lineRule="exact"/></w:pPr></w:p>')
    return "".join(out)


# ════════════════════════════════════════════════════════════════════════════
# CONTENT
# ════════════════════════════════════════════════════════════════════════════
TITLE = "When Validation Lies: Diagnosing and Fixing Model Selection Failures Under Real Hospital Shift"

ABSTRACT = (
    "Clinical machine learning models degrade when deployed across hospitals, and a "
    "large literature proposes methods to close that gap. We set out to build one: "
    "Physiology-Constrained Learning (PCL), which augments masked-prediction pretraining "
    "of a clinical time-series transformer with differentiable penalties enforcing known "
    "physiological relationships. PCL initially appeared to improve zero-shot cross-hospital "
    "sepsis prediction substantially. Systematic re-evaluation showed that the improvement "
    "did not survive proper controls: it was attributable to a label-definition confound, a "
    "pretraining leak that broke the zero-shot premise, hyperparameter selection performed on "
    "out-of-distribution data, and run-to-run variance exceeding the reported effect. "
    "Correcting all four eliminated the gain. That investigation surfaced a separate and more "
    "broadly relevant problem. Selecting hyperparameters on held-out source-hospital data, the "
    "standard practice, is unreliable under real hospital shift: across a constraint-weight "
    "sweep, source validation AUROC varied by roughly one point while target AUROC varied by "
    "eleven, and source-validation selection performed no better than choosing at random. "
    "Scoring candidate configurations with masked-reconstruction error on unlabeled target data "
    "reduced mean selection regret from 0.052 to 0.012. Results span three real hospital systems "
    "and roughly 230,000 ICU stays across three seeds."
)

KEYWORDS = ("domain generalization, clinical machine learning, model selection, "
            "distribution shift, hyperparameter tuning, ICU prediction")

FIGURES = {"rIdFig1": "fig1_validation_gap.png",
           "rIdFig2": "fig2_design.png",
           "rIdFig3": "fig3_regret.png"}

BODY = []
A = BODY.append

# ── I. Introduction ──────────────────────────────────────────────────────────
A(para("Introduction", "Heading1"))
A(para(
    "Machine learning models trained at one hospital routinely lose accuracy when deployed at "
    "another. The causes are well documented: measurement devices differ, laboratory analyzers are "
    "calibrated independently, and local ordering habits induce site-specific correlations between "
    "how data is recorded and what is being predicted. A model that latches onto these artifacts "
    "rather than underlying physiology will not transfer [12], [13], [23]-[25]."))
A(para(
    "Most proposed remedies either require labeled data from the target hospital, which is "
    "precisely what is unavailable before deployment, or rely on having many training environments, "
    "which clinical settings rarely provide [7]-[10], [22]. This motivates methods that require neither. We "
    "pursued one such method, encoding known physiological relationships as pretraining constraints, "
    "on the premise that physiology is shared across hospitals even when measurement is not."))
A(para(
    "This paper reports what happened when that method was tested rigorously, and what the testing "
    "revealed. Our contributions are: (1) a case study in which an apparently strong cross-hospital "
    "result dissolved under four separable controls, each of which is easy to reproduce accidentally; "
    "(2) a quantified demonstration, at full scale on three real hospital systems, that "
    "source-hospital validation is an unreliable criterion for selecting hyperparameters intended for "
    "deployment elsewhere; and (3) evidence that a cheap, label-free alternative, scoring candidate "
    "configurations by reconstruction error on unlabeled target data, recovers most of the lost "
    "performance. The third result is not a new algorithm; it confirms, in a high-stakes clinical "
    "setting and at realistic scale, a phenomenon that the domain-adaptation literature has "
    "benchmarked in other contexts."))

# ── II. Background ───────────────────────────────────────────────────────────
A(para("Background", "Heading1"))
A(para(
    "Hyperparameter selection normally relies on a held-out split of the training distribution. This "
    "is justified by exchangeability: validation data is drawn from the same distribution as the "
    "deployment data, so validation performance estimates deployment performance. Cross-hospital "
    "deployment violates that assumption by construction, since the deployment distribution is a "
    "different hospital. Whether source validation still ranks configurations correctly under such "
    "shift is an empirical question, and one that is rarely checked."))
A(para(
    "The domain-generalization literature has raised this concern before. Benchmark studies have shown that measured differences between "
    "generalization algorithms depend heavily on the model selection criterion [6], that "
    "unsupervised scorers on unlabeled target data are a viable alternative [11], [19], and that "
    "reported adaptation gains often shrink under standardized protocols [20], [21]. Classical "
    "theory ties target error to a source-target divergence that source validation cannot "
    "observe [18]. "
    "Our contribution here is not the idea but the evidence: we test it on real "
    "multi-hospital electronic health record data at full scale, where the shift is genuine rather "
    "than simulated."))

# ── III. PCL case study ──────────────────────────────────────────────────────
A(para("Motivating Case Study: Physiology-Constrained Learning", "Heading1"))
A(para("Method and Setup", "Heading2"))
A(para(
    "PCL pretrains a six-layer transformer [15] (d = 256, 8 heads, about 3.2M parameters) on 48-hour, "
    "hourly-binned ICU windows using masked-value prediction [16], adding a differentiable penalty for "
    "violating three physiological relationships among the model's reconstructed outputs: the mean "
    "arterial pressure identity, the Henderson-Hasselbalch equation, and the Severinghaus oxygen "
    "dissociation relation. The penalty weight is denoted by the constraint weight. Constraints are "
    "applied only at timesteps where all required variables were observed and at least one was "
    "masked, so the model must impute from cross-variable structure rather than copy its input."))
A(picture("rIdFig2", 2926080, 1322185, "Figure 2"))
A(para(
    "Study design. Models are trained and validated on PhysioNet Site A only. Site B shares the "
    "collection protocol; MIMIC-IV and eICU-CRD are independent EHR systems. No target-site data, "
    "labeled or unlabeled, is used during training, and no selection decision uses target labels.",
    "figurecaption", keep=True))
A(para(
    "Table I summarizes the cohorts. Sepsis prevalence differs substantially across systems, which "
    "is itself part of the shift being studied."))
A(para("Evaluation cohorts after filtering to adult stays of at least 24 hours.", "tablehead"))
A(table([
    ["Cohort", "Stays", "Sepsis+", "Role"],
    ["PhysioNet Site A", "10,381", "3.3%", "train / val"],
    ["PhysioNet Site B", "14,779", "2.1%", "target"],
    ["MIMIC-IV", "74,607", "30.9%", "target"],
    ["eICU-CRD", "130,446", "8.8%", "target"],
], [1500, 1000, 1100, 1300]))
A(para(
    "Models are trained on PhysioNet Challenge 2019 Site A [3] and evaluated zero-shot on three held-out "
    "cohorts: PhysioNet Site B (same collection protocol), MIMIC-IV [4] (74,607 stays), and eICU-CRD [5] "
    "(130,446 stays across many hospitals). After filtering, the pooled cohort is 230,213 ICU stays. "
    "The task is sepsis onset prediction. Seventeen variables are used, comprising vital signs, "
    "arterial blood gases, and routine chemistry and hematology."))

A(para("Are the Constraints Non-Trivial?", "Heading2"))
A(para(
    "Before evaluating transfer we audited whether the constraints carry signal at all. In raw "
    "pre-normalization data the mean arterial pressure identity is violated beyond 3 mmHg in 44.0% of "
    "computable hours at Site A, 54.2% in MIMIC-IV, and 64.7% in eICU-CRD; Henderson-Hasselbalch is "
    "violated beyond 0.05 pH units in 13.8%, 23.8%, and 18.5% respectively. The constraints are "
    "therefore not tautologies: independent sensors and analyzers genuinely disagree, and they "
    "disagree more in the external systems than in the training distribution."))
A(para(
    "Constraint availability is highly uneven, however. The blood-pressure identity is computable at "
    "70 to 80% of stay-hours, but the acid-base constraints are computable at only 0.6 to 1.8% of "
    "hours because arterial blood-gas panels are drawn infrequently. Measured on the loss actually "
    "optimized, the Henderson-Hasselbalch term contributes roughly one twenty-seventh the magnitude "
    "of the blood-pressure term, so the objective is dominated by a single constraint even though "
    "three are nominally active."))
A(para(
    "That unevenness has a consequence we did not anticipate, and it matters for any use of the "
    "violation score as a site-level signal. Because the score averages whichever constraints happen "
    "to be computable, and because the constraints occupy very different magnitude ranges, its "
    "per-site mean partly tracks laboratory ordering practice rather than physiological consistency. "
    "PhysioNet Site A charts bicarbonate in 8.1% of stay-hours against 0.16% at Site B, a fifty-fold "
    "difference, which makes the acid-base constraints computable 14.7 times more often at Site A and "
    "the oxygen-dissociation constraint 3.2 times more often. Table II shows the effect. Measured on "
    "raw data with no model involved, the blood-pressure residual is essentially identical at the two "
    "sites, while the Severinghaus residual is four times larger in scale and differs by a factor of "
    "8.6 between them. Averaging under each site’s own constraint mix reproduces the observed "
    "cross-site gap in violation without any model: 0.032 at Site A against 0.010 at Site B."))
A(para("Per-constraint residuals on raw data (PhysioNet Sites A and B, first 4,000 stays each). "
       "MAP is comparably available at both sites and shows no site difference; the site gap in mean "
       "violation is produced by the sparser, larger-magnitude Severinghaus term.", "tablehead"))
A(table([
    ["Constraint", "Site A", "Site B", "Scale vs MAP"],
    ["MAP identity", "0.0188", "0.0206", "1.00"],
    ["Henderson-Hasselbalch", "0.0008", "0.0004", "0.04"],
    ["Severinghaus", "0.0773", "0.0090", "4.12"],
], [1900, 900, 900, 1050]))
A(para(
    "We report this because it bounds what the violation score can be used for. It remains a valid "
    "training signal, since the penalty is applied per timestep where the constraint is computable and "
    "is never compared across sites. But its per-site average is confounded with measurement "
    "availability, so it should not be interpreted as a measure of how physiologically consistent a "
    "hospital’s data is, nor used as a site-level distribution-shift statistic without first "
    "standardizing each constraint separately. A further caveat applies to Severinghaus on PhysioNet "
    "specifically: the dataset does not record arterial oxygen partial pressure, so it is "
    "reconstructed by inverting the same relation the constraint enforces, as noted below."))
A(para("Four Confounds", "Heading2"))
A(para(
    "Under our initial evaluation PCL outperformed empirical risk minimization (ERM) at every "
    "held-out site. Systematic re-testing identified four distinct problems, each of which "
    "independently inflated the apparent gain."))
A(para(
    "Label-definition shift. Training labels were clinical Sepsis-3, while external evaluation used "
    "sepsis labels derived from administrative ICD codes. Cross-site differences therefore conflated "
    "domain shift with a change in the definition of the outcome. We re-derived Sepsis-3 directly on "
    "MIMIC-IV and eICU using SOFA scoring anchored to a suspected-infection window, following the "
    "standard reference implementation [1], [2], yielding positive rates of 30.9% and 8.8% respectively; clinical and claims-based sepsis definitions are known to diverge substantially [26].",
    "bulletlist"))
A(para(
    "Pretraining leak. PCL pretrained on unlabeled data from all sites while ERM pretrained on the "
    "source site alone. Although pretraining is self-supervised and uses no labels, exposure to "
    "target-site inputs breaks the zero-shot premise and constitutes unintended domain adaptation.",
    "bulletlist"))
A(para(
    "Contaminated hyperparameter selection. The constraint weight was selected using performance on "
    "a held-out site rather than source validation data, so the reported zero-shot numbers were "
    "chosen with knowledge of the test distribution.", "bulletlist"))
A(para(
    "Partial circularity. The source dataset does not record arterial oxygen partial pressure "
    "directly; it was reconstructed from oxygen saturation by inverting the same Severinghaus "
    "relation the model was penalized for violating, making that constraint partly self-referential "
    "on training data.", "bulletlist"))

A(para("Result After Correction", "Heading2"))
A(para(
    "With all four corrected, PCL no longer improved cross-hospital transfer. Table III reports the "
    "corrected comparison over three seeds. PCL is below ERM at every site, and a gradient-boosted "
    "tree [14] trained on simple per-stay summary statistics matches or exceeds both neural models, which "
    "further questions the architecture. A group-distributionally-robust baseline, once corrected for "
    "class imbalance, is the strongest neural configuration tested."))

A(para("Corrected cross-hospital sepsis AUROC (mean over 3 seeds). "
       "Site B, MIMIC-IV, and eICU are zero-shot. XGBoost uses per-stay summary features.",
       "tablehead"))
A(table([
    ["Method", "Source", "Site B", "MIMIC-IV", "eICU"],
    ["ERM", "0.818", "0.701", "0.649", "0.592"],
    ["PCL", "0.798", "0.655", "0.622", "0.572"],
    ["Group DRO", "0.822", "0.717", "0.688", "0.628"],
    ["XGBoost", "\u2014", "0.754", "0.654", "0.660"],
], [1150, 800, 830, 1090, 830]))
A(para(
    "Expanding the input from the original nine physiological channels to seventeen, adding "
    "temperature, respiratory rate, and routine chemistry and hematology, improved external transfer "
    "for both methods and was retained throughout. Two unit-harmonization defects surfaced while "
    "doing so and are worth recording, since either would silently damage transfer: MIMIC-IV charts "
    "temperature predominantly in Fahrenheit, and eICU-CRD records it in nurse charting rather than "
    "the periodic vitals table. Left uncorrected, temperature was observed for 82% of source hours "
    "but only 6% of target hours, an availability gap that penalizes any model relying on it."))
A(para(
    "Two further observations support the negative result. Invariant risk minimization with 30 "
    "hospital environments drawn from eICU performed no better than ERM (0.788 versus 0.792), "
    "indicating the failure is not confined to the low-environment regime. Separately, run-to-run "
    "variability across seeds reached 4 to 7 AUROC points, which exceeds most of the differences "
    "originally reported as method effects."))

# ── IV. Validation problem ───────────────────────────────────────────────────
A(para("The Validation Problem, Quantified", "Heading1"))
A(para(
    "Correcting the third confound raised a question that outlives PCL itself. If the constraint "
    "weight must be chosen without touching target data, does source-hospital validation choose it "
    "well? We swept the constraint weight over six values, recorded source validation AUROC and true "
    "AUROC at each held-out site, and compared the configuration that validation would select against "
    "the best configuration available."))
A(para(
    "Table IV shows the disconnect. Source validation AUROC spans about one point across the entire "
    "sweep, while target AUROC spans roughly eleven. Validation is nearly flat precisely where the "
    "target performance differences are largest, so it carries little information about which "
    "configuration will deploy well. In this seed, source validation selects the weakest "
    "configuration for every external site."))
A(para("Source validation AUROC versus true target AUROC across the constraint-weight "
       "sweep (single seed). Validation spans ~0.01; target spans ~0.11.", "tablehead"))
A(table([
    ["Weight", "Source", "Site B", "MIMIC-IV", "eICU"],
    ["0.0 (ERM)", "0.835", "0.692", "0.601", "0.548"],
    ["0.1", "0.831", "0.737", "0.691", "0.637"],
    ["0.5", "0.825", "0.750", "0.695", "0.662"],
    ["1.0", "0.827", "0.675", "0.645", "0.585"],
    ["2.0", "0.820", "0.653", "0.633", "0.562"],
    ["5.0", "0.828", "0.597", "0.608", "0.575"],
], [1000, 850, 880, 1090, 880]))
A(picture("rIdFig1", 2926080, 2354788, "Figure 1"))
A(para(
    "Source-hospital validation AUROC versus true zero-shot target AUROC, one point per "
    "(constraint weight, target site). Points scatter almost vertically: validation varies by 0.015 "
    "while target performance varies by 0.20, and the trend is mildly downward rather than upward. "
    "The configuration source validation selects is among the weakest at every external site.",
    "figurecaption", keep=True))
A(para(
    "We stress that the best-performing weight is not stable across seeds: in a second seed the "
    "unconstrained model was strongest at two of three sites. This instability is itself part of the "
    "finding. It means the practical question is not which weight is universally best, but whether "
    "any available selection criterion can identify a good one for a given deployment."))

# ── V. Fix ───────────────────────────────────────────────────────────────────
A(para("Unsupervised Target-Side Selection", "Heading1"))
A(para(
    "We therefore evaluated selection criteria that use unlabeled target data only. For each trained "
    "configuration and each target site, we computed four label-free scores in a single "
    "inference-only pass: masked-reconstruction error, physiological constraint violation, predictive "
    "entropy, and distance between source and target representation centroids. Each score is "
    "lower-is-better and is used to rank configurations."))
A(para(
    "We report two metrics. Spearman rank correlation measures whether a score orders configurations "
    "the way true target AUROC does; a more negative value is better. Selection regret is the true "
    "target AUROC of the configuration a criterion selects minus that of the best available "
    "configuration, so zero is optimal. Both are computed over three seeds and three target sites, "
    "and are reported in Table V."))
A(para("Label-free selection criteria. Rank correlation with true target AUROC "
       "(more negative is better) and mean selection regret (closer to zero is better), "
       "over 3 seeds and 3 sites.", "tablehead"))
A(table([
    ["Selection criterion", "Rank corr.", "Mean regret"],
    ["Reconstruction error", "-0.459", "-0.012"],
    ["Constraint violation", "-0.426", "-0.016"],
    ["Representation distance", "+0.002", "-0.030"],
    ["Predictive entropy", "+0.150", "-0.076"],
    ["Random selection", "\u2014", "-0.043"],
    ["Source validation", "\u2014", "-0.052"],
], [2500, 1200, 1200]))
A(picture("rIdFig3", 2926080, 1730078, "Figure 3"))
A(para(
    "Mean selection regret by criterion, averaged over three seeds and three target sites. Lower is "
    "better. Source validation (the standard practice) and reconstruction error (the proposed "
    "substitute) are highlighted.", "figurecaption", keep=True))
A(para(
    "Table VI reports the per-seed rank correlations, since a criterion that only works on one seed "
    "is of no practical use. Reconstruction error and constraint violation are negative in all three "
    "seeds, though the strength decays; entropy and representation distance are unstable in sign."))
A(para("Rank correlation with true target AUROC by seed (more negative is better).", "tablehead"))
A(table([
    ["Criterion", "Seed 42", "Seed 43", "Seed 44"],
    ["Reconstruction error", "-0.600", "-0.521", "-0.255"],
    ["Constraint violation", "-0.608", "-0.391", "-0.280"],
    ["Representation distance", "+0.102", "-0.040", "-0.057"],
    ["Predictive entropy", "+0.216", "+0.170", "+0.065"],
], [1900, 1000, 1000, 1000]))
A(para(
    "Three findings follow. First, source-hospital validation performs no better than selecting a "
    "configuration at random, and in two of three seeds it is worse; the difference from random is "
    "smaller than seed-to-seed variability, so we claim only that it is uninformative, not reliably "
    "harmful. Second, reconstruction error on unlabeled target data reduces mean regret from 0.052 to "
    "0.012, recovering most of the gap to oracle selection, and its advantage is consistent in sign "
    "across all three seeds. Third, predictive entropy and representation distance are not viable "
    "substitutes here; entropy is actively misleading, selecting the worst configuration in every "
    "seed."))
A(para(
    "Notably, the physiological constraint violation score does not outperform generic reconstruction "
    "error. The useful signal is that the model reconstructs target data poorly when it is configured "
    "badly, an observation consistent with denoising-autoencoder pretraining [17], and domain-specific physiological knowledge adds nothing on top of that. This is a "
    "negative result for our original method and a positive one for practitioners, since the "
    "effective criterion requires no domain expertise, no target labels, no additional parameters, "
    "and one inference pass."))

# ── VI. Limitations ──────────────────────────────────────────────────────────
A(para("Practical Recommendation", "Heading2"))
A(para(
    "The procedure we would recommend is deliberately unglamorous. Train the candidate "
    "configurations as usual. Before deployment, collect unlabeled data from the target hospital, "
    "which requires no annotation effort and no clinical review, and run one forward pass per "
    "candidate to obtain its masked-reconstruction error on that data. Select the configuration with "
    "the lowest error. The cost is a single inference pass per candidate, no additional parameters, "
    "and no target labels, and in our experiments it recovered roughly three quarters of the gap "
    "between standard practice and oracle selection."))
A(para(
    "We would also encourage reporting seed variability as a matter of course. In this study "
    "run-to-run variation reached 4 to 7 AUROC points, which is larger than most published "
    "cross-hospital improvements we are aware of, including the improvement we originally believed "
    "we had found. A single-seed comparison at this scale is not informative regardless of how large "
    "the reported difference is."))
A(para("Limitations", "Heading1"))
A(para(
    "Our evaluation covers a single downstream task, sepsis onset prediction, and a single "
    "architecture; standard multi-task ICU benchmarks [27] and the wider sepsis-prediction "
    "literature [28], [29] offer natural extensions. Whether the selection result holds for other "
    "clinical targets or model families is "
    "untested. Three seeds is a small sample given the 4 to 7 point run-to-run variance we measured, "
    "and secondary comparisons should be read with that in mind. Our external labels are re-derived "
    "Sepsis-3 rather than chart review, so residual label noise remains. Finally, we studied ICU "
    "time-series data from three systems; extension to other clinical domains, and to shifts that are "
    "not primarily driven by measurement practice, is future work."))

# ── VII. Conclusion ──────────────────────────────────────────────────────────
A(para("Conclusion", "Heading1"))
A(para(
    "We built a physiologically-constrained representation-learning method, found that its apparent "
    "cross-hospital advantage did not survive correcting four confounds, and in the process "
    "identified a problem that matters more than the method did. Selecting hyperparameters on "
    "source-hospital validation data, which is standard practice, carries little information about "
    "performance at a new hospital: in our sweep it was no better than random. Scoring candidate "
    "configurations by reconstruction error on unlabeled target data is cheap, requires no labels or "
    "domain knowledge, and recovered most of the achievable performance. For teams deploying clinical "
    "models across sites, that substitution is a small change to standard practice with a measurable "
    "payoff."))

# ── Acknowledgment ───────────────────────────────────────────────────────────
A(para("Acknowledgment", "Heading5"))
A(para(
    "The authors thank the PhysioNet team and the MIMIC-IV and eICU-CRD Collaborative Research "
    "Database maintainers for curating and providing access to the credentialed datasets used "
    "in this study."))

# ── References ───────────────────────────────────────────────────────────────
A(para("References", "Heading5"))
REFS = [
    "M. Singer et al., \u201cThe Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3),\u201d JAMA, vol. 315, no. 8, pp. 801\u2013810, 2016.",
    "C. W. Seymour et al., \u201cAssessment of clinical criteria for sepsis,\u201d JAMA, vol. 315, no. 8, pp. 762\u2013774, 2016.",
    "M. A. Reyna et al., \u201cEarly prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019,\u201d Crit. Care Med., vol. 48, no. 2, pp. 210\u2013217, 2019.",
    "A. E. W. Johnson et al., \u201cMIMIC-IV, a freely accessible electronic health record dataset,\u201d Sci. Data, vol. 10, no. 1, 2023.",
    "T. J. Pollard et al., \u201cThe eICU Collaborative Research Database,\u201d Sci. Data, vol. 5, 2018.",
    "I. Gulrajani and D. Lopez-Paz, \u201cIn search of lost domain generalization,\u201d in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.",
    "M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz, \u201cInvariant risk minimization,\u201d arXiv:1907.02893, 2019.",
    "E. Rosenfeld, P. Ravikumar, and A. Risteski, \u201cThe risks of invariant risk minimization,\u201d in Proc. ICLR, 2021.",
    "S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, \u201cDistributionally robust neural networks for group shifts,\u201d in Proc. ICLR, 2020.",
    "Y. Ganin et al., \u201cDomain-adversarial training of neural networks,\u201d J. Mach. Learn. Res., vol. 17, no. 59, pp. 1\u201335, 2016.",
    "Y. Lalou, T. Gnassounou, A. Collas, A. de Mathelin, O. Kachaiev, A. Odonnat, A. Gramfort, T. Moreau, and R. Flamary, \u201cSKADA-Bench: benchmarking unsupervised domain adaptation methods with realistic validation on diverse modalities,\u201d arXiv:2407.11676, 2024.",
    "S. G. Finlayson et al., \u201cThe clinician and dataset shift in artificial intelligence,\u201d N. Engl. J. Med., vol. 385, no. 3, pp. 283\u2013286, 2021.",
    "A. Wong et al., \u201cExternal validation of a widely implemented proprietary sepsis prediction model in hospitalized patients,\u201d JAMA Intern. Med., vol. 181, no. 8, pp. 1065\u20131070, 2021.",
    "T. Chen and C. Guestrin, \u201cXGBoost: a scalable tree boosting system,\u201d in Proc. ACM SIGKDD, 2016, pp. 785\u2013794.",
    "A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017.",
    "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL-HLT, 2019, pp. 4171–4186.",
    "P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Extracting and composing robust features with denoising autoencoders,” in Proc. Int. Conf. Mach. Learn. (ICML), 2008, pp. 1096–1103.",
    "S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan, “A theory of learning from different domains,” Mach. Learn., vol. 79, no. 1–2, pp. 151–175, 2010.",
    "K. You, X. Wang, M. Long, and M. I. Jordan, “Towards accurate model selection in deep unsupervised domain adaptation,” in Proc. ICML, 2019, pp. 7124–7133.",
    "K. Musgrave, S. Belongie, and S.-N. Lim, “Unsupervised domain adaptation: a reality check,” arXiv:2111.15672, 2021.",
    "P. W. Koh et al., “WILDS: a benchmark of in-the-wild distribution shifts,” in Proc. ICML, 2021, pp. 5637–5664.",
    "B. Sun and K. Saenko, “Deep CORAL: correlation alignment for deep domain adaptation,” in Proc. ECCV Workshops, 2016, pp. 443–450.",
    "J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, “Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study,” PLoS Med., vol. 15, no. 11, e1002683, 2018.",
    "J. Futoma, M. Simons, T. Panch, F. Doshi-Velez, and L. A. Celi, “The myth of generalisability in clinical research and machine learning in health care,” Lancet Digit. Health, vol. 2, no. 9, pp. e489–e492, 2020.",
    "B. Nestor et al., “Feature robustness in non-stationary health records,” in Proc. Mach. Learn. Healthc. Conf. (MLHC), 2019, pp. 381–408.",
    "C. Rhee et al., “Incidence and trends of sepsis in US hospitals using clinical vs claims data, 2009–2014,” JAMA, vol. 318, no. 13, pp. 1241–1249, 2017.",
    "H. Harutyunyan, H. Khachatrian, D. C. Kale, G. Ver Steeg, and A. Galstyan, “Multitask learning and benchmarking with clinical time series data,” Sci. Data, vol. 6, no. 1, 96, 2019.",
    "L. M. Fleuren et al., “Machine learning for the prediction of sepsis: a systematic review and meta-analysis of diagnostic test accuracy,” Intensive Care Med., vol. 46, no. 3, pp. 383–400, 2020.",
    "M. Moor, B. Rieck, M. Horn, C. R. Jutzeler, and K. Borgwardt, “Early prediction of sepsis in the ICU using machine learning: a systematic review,” Front. Med., vol. 8, 607952, 2021.",
]
for r in REFS:
    A(f'<w:p><w:pPr><w:pStyle w:val="references"/></w:pPr>{runs(r)}</w:p>')


# ════════════════════════════════════════════════════════════════════════════
# ASSEMBLE
# ════════════════════════════════════════════════════════════════════════════
def main():
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    with zipfile.ZipFile(TPL) as z:
        z.extractall(WORK)

    dpath = os.path.join(WORK, "word", "document.xml")
    xml = open(dpath, encoding="utf-8").read()
    bstart = xml.index("<w:body>") + len("<w:body>")
    bend = xml.rindex("</w:body>")
    body = xml[bstart:bend]

    blocks = [m.group(0) for m in
              re.finditer(r"<w:(p|tbl)(?: [^>]*)?>.*?</w:\1>", body, re.S)]
    tail = body[body.rindex(blocks[-1]) + len(blocks[-1]):]  # trailing sectPr

    # The template's author region is a FOUR-COLUMN grid sized for up to six
    # authors. Reusing it for a single author left the name stranded in column one
    # of four with ~2in of dead space below. Instead we rebuild the header as one
    # full-width (single-column) section holding the title and author block, ended
    # by a continuous section break; everything after it falls into the
    # document-level two-column section (abstract onward), which is the IEEE layout.
    doc_sect = re.search(r"<w:sectPr.*?</w:sectPr>", tail, re.S).group(0)
    pg = "".join(re.findall(r"<w:pgSz[^/]*/>|<w:pgMar[^/]*/>", doc_sect))
    head_sect = (f'<w:sectPr><w:type w:val="continuous"/>{pg}'
                 f'<w:cols w:space="36pt"/></w:sectPr>')

    def titled(text, style):
        return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{runs(text)}</w:p>'

    # Two authors side by side, IEEE-style, in a borderless table so the columns
    # stay balanced regardless of name length.
    AUTHORS = [
        ["Nikhil Tamvada", "Independent Researcher",
         "Pleasanton, CA, USA", "nikhiltamvada@gmail.com"],
        ["Weizhi Lin", "San Jose State University",
         "San Jose, CA, USA", "weizhi.lin@sjsu.edu"],
    ]

    def author_cell(lines, w):
        body = runs(lines[0])
        for l in lines[1:]:
            body += '<w:r><w:br/></w:r>' + runs(l)
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="Author"/></w:pPr>{body}</w:p></w:tc>')

    cw = 4700
    author_tbl = (
        f'<w:tbl><w:tblPr><w:tblW w:w="{cw*2}" w:type="dxa"/>'
        '<w:jc w:val="center"/><w:tblBorders>'
        + "".join(f'<w:{b} w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
                  for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + '</w:tblBorders></w:tblPr>'
        f'<w:tblGrid><w:gridCol w:w="{cw}"/><w:gridCol w:w="{cw}"/></w:tblGrid>'
        '<w:tr>' + "".join(author_cell(a, cw) for a in AUTHORS) + '</w:tr></w:tbl>')
    head = [titled(TITLE, "papertitle"), author_tbl]
    # Paragraph carrying the section break that closes the full-width region.
    head.append(f'<w:p><w:pPr><w:pStyle w:val="Author"/>{head_sect}</w:pPr></w:p>')

    def styled(text, style, prefix):
        return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
                f'{runs(prefix, bold=True, italic=True)}{runs(text)}</w:p>')

    head.append(styled(ABSTRACT, "Abstract", "Abstract\u2014"))
    head.append(styled(KEYWORDS, "Keywords", "Keywords\u2014"))

    # The template's two-column definition lived in a section break inside the
    # placeholder body we just removed. The final (document-level) sectPr now
    # governs everything from the abstract onward, so it must carry the
    # two-column layout IEEE requires; the title and author blocks keep their own
    # earlier section breaks and remain full width.
    def force_two_col(sect):
        if "<w:cols" in sect:
            return re.sub(r"<w:cols[^/]*/>", '<w:cols w:num="2" w:space="18pt"/>', sect)
        return sect.replace("</w:sectPr>", '<w:cols w:num="2" w:space="18pt"/></w:sectPr>')

    tail = re.sub(r"<w:sectPr.*?</w:sectPr>",
                  lambda m: force_two_col(m.group(0)), tail, flags=re.S)

    new_body = "".join(head) + "".join(BODY) + tail
    open(dpath, "w", encoding="utf-8").write(xml[:bstart] + new_body + xml[bend:])

    # ── figure plumbing: media file + relationship + content-type override ──
    media = os.path.join(WORK, "word", "media")
    os.makedirs(media, exist_ok=True)
    rels_p = os.path.join(WORK, "word", "_rels", "document.xml.rels")
    rels = open(rels_p, encoding="utf-8").read()
    for rid, src in FIGURES.items():
        srcp = os.path.join(HERE, src)
        if not os.path.exists(srcp):
            raise SystemExit(f"{src} missing — run make_figure.py first")
        shutil.copy(srcp, os.path.join(media, src))
        if rid not in rels:
            rels = rels.replace("</Relationships>",
                f'<Relationship Id="{rid}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="media/{src}"/></Relationships>')
    open(rels_p, "w", encoding="utf-8").write(rels)

    ct_p = os.path.join(WORK, "[Content_Types].xml")
    ct = open(ct_p, encoding="utf-8").read()
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Types ", "<Types ", 1).replace(
            "</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
        open(ct_p, "w", encoding="utf-8").write(ct)

    # The wp:/a:/pic: prefixes used by the drawing must be declared on w:document.
    # Add ONLY the ones the template does not already declare -- re-adding an
    # existing xmlns makes the part malformed (duplicate attribute).
    xml2 = open(dpath, encoding="utf-8").read()
    m = re.search(r"<w:document\b[^>]*>", xml2)
    tag = m.group(0)
    need = {
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    }
    add = "".join(f' xmlns:{k}="{v}"' for k, v in need.items()
                  if f"xmlns:{k}=" not in tag)
    if add:
        xml2 = xml2.replace(tag, tag[:-1] + add + ">", 1)
        open(dpath, "w", encoding="utf-8").write(xml2)

    if os.path.exists(OUT):
        os.remove(OUT)
    zf = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
    for root, _, files in os.walk(WORK):
        for f in files:
            p = os.path.join(root, f)
            zf.write(p, os.path.relpath(p, WORK))
    zf.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
