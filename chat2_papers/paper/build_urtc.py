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
import struct
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


def png_size(path):
    """(width, height) in pixels, straight out of the IHDR chunk."""
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


COL_EMU = 2926080  # 3.2in: fits the IEEE column with a little side margin


def figure(rid, name, scale=1.0):
    """Column-width image, height derived from the file's own aspect ratio.
    Hand-set heights drift out of sync whenever a figure is redrawn, which
    renders it squashed; deriving the height removes that failure mode.
    `scale` narrows a schematic that does not need the full column."""
    w, h = png_size(os.path.join(HERE, FIGURES[rid]))
    wid = int(round(COL_EMU * scale))
    return picture(rid, wid, int(round(wid * h / w)), name)


def eq(parts):
    """Centered display equation, built from (text, italic, subscript) runs.

    NOTE: the template's `equation` style maps its runs to the Symbol font, so
    "L = L_mask" comes out as "Λ = Λ_μασκ". The equation therefore uses a plain
    centered paragraph with explicitly-fonted runs instead of that style.
    """
    body = ""
    for text, italic, sub in parts:
        rpr = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'
               + ("<w:i/>" if italic else "")
               + ('<w:vertAlign w:val="subscript"/>' if sub else ""))
        body += (f'<w:r><w:rPr>{rpr}</w:rPr>'
                 f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>')
    return ('<w:p><w:pPr><w:spacing w:before="120" w:after="120" w:line="240" '
            'w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>' + body + '</w:p>')


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
#
# Structure follows the reviewer's outline: the paper answers one question --
# how reliable is source-domain validation under real hospital shift -- and PCL
# is the experimental vehicle used to answer it, not a claimed contribution.
# ════════════════════════════════════════════════════════════════════════════
TITLE = "When Validation Lies: Diagnosing and Fixing Model Selection Failures Under Real Hospital Shift"

ABSTRACT = (
    "Deploying a clinical machine learning model at a new hospital requires choosing, before any "
    "outcome at that hospital is observable, which of the configurations one has trained will "
    "transfer best. Standard practice makes that choice on held-out source-hospital data. We ask how "
    "reliable that practice is under real hospital distribution shift. Our experimental vehicle is "
    "Physiology-Constrained Learning (PCL), which augments masked-prediction pretraining of a clinical "
    "time-series transformer with differentiable penalties enforcing known physiological relationships; "
    "its constraint weight is a hyperparameter whose value strongly affects cross-hospital performance. "
    "PCL initially appeared to improve zero-shot cross-hospital sepsis prediction substantially. "
    "Systematic re-evaluation showed that the improvement did not survive proper controls: it was "
    "attributable to four separable evaluation confounds, and correcting them eliminated the gain and "
    "exposed the selection problem directly. Across the sweep, source validation AUROC varied by "
    "roughly one point "
    "while target AUROC varied by eleven, and source-validation selection performed no better than "
    "choosing at random. Scoring candidate configurations with masked-reconstruction error on unlabeled "
    "target data reduced mean selection regret from 0.052 to 0.012. Results span three real hospital "
    "systems and roughly 230,000 ICU stays across three seeds."
)

KEYWORDS = ("domain generalization, clinical machine learning, model selection, "
            "distribution shift, hyperparameter tuning, ICU prediction")

FIGURES = {"rIdFig0": "fig0_overview.png",
           "rIdFig1": "fig1_validation_gap.png",
           }

BODY = []
A = BODY.append

# ── I. Introduction (Background merged in) ───────────────────────────────────
A(para("Introduction", "Heading1"))
A(para(
    "Machine learning models trained at one hospital routinely lose accuracy when deployed at "
    "another. The causes are well documented: measurement devices differ, laboratory analyzers are "
    "calibrated independently, and local ordering habits induce site-specific correlations between "
    "how data is recorded and what is being predicted. A model that latches onto these artifacts "
    "rather than underlying physiology will not transfer [1]-[3]."))
A(para(
    "Reliable deployment therefore demands more than a training procedure that is robust in "
    "principle. It demands choosing, among the configurations actually trained, the one that will "
    "perform best at a hospital whose data has never been seen. That choice must be made before a "
    "single outcome at the target hospital is observable, and whatever robustness a method offers is "
    "realized only if the selection step can find it. Standard practice makes the choice on a held-out "
    "split of the training distribution, justified by exchangeability: validation data is drawn from "
    "the same distribution as the deployment data, so validation performance estimates deployment "
    "performance. Cross-hospital deployment violates that assumption by construction. Whether source "
    "validation still ranks configurations correctly under such shift is an empirical question, and "
    "one that is rarely checked."))
A(para(
    "This paper asks that question directly: how reliable is conventional source-domain validation "
    "for model selection under real hospital distribution shift, and is there a practical alternative "
    "that requires no target labels? Answering it requires genuine rather than simulated shift, and a "
    "hyperparameter whose choice demonstrably changes deployment performance."))
A(para(
    "We answer it through a concrete experimental framework. Physiology-Constrained Learning (PCL) "
    "augments masked-prediction pretraining of a clinical time-series transformer with differentiable "
    "penalties for violating known physiological relationships, on the intuition that physiology is "
    "shared across hospitals even when measurement practice is not. PCL supplies exactly what the "
    "selection question needs: a single hyperparameter, the constraint weight, that spans a wide range "
    "of cross-hospital behavior, and a family of trained configurations that any candidate criterion "
    "can be asked to rank. PCL initially appeared to improve zero-shot cross-hospital sepsis "
    "prediction substantially; a rigorous re-evaluation identified four separable evaluation confounds "
    "that together accounted for the entire apparent gain. That re-evaluation is what exposed the "
    "selection problem, because the third confound, choosing the constraint weight on "
    "out-of-distribution data, is a confound only if the legitimate alternative works."))
A(para(
    "That alternative proves unreliable. Our contributions answer the central question in order. "
    "First, we show that "
    "conventional source-domain validation is unreliable for model selection under hospital "
    "distribution shift: across a constraint-weight sweep on three real hospital systems, source "
    "validation AUROC varied by roughly one point while target AUROC varied by eleven, and the "
    "selected configuration was among the weakest at every external site. Second, we use the PCL study "
    "to demonstrate concretely how a seemingly promising cross-hospital improvement disappears once "
    "four common evaluation confounds are corrected, each easy to reproduce accidentally. Third, we "
    "show empirically that reconstruction error on unlabeled target data is an effective selection "
    "criterion, cutting mean selection regret from 0.052 to 0.012. The third result is not a new "
    "algorithm; it confirms, in a high-stakes clinical setting and at realistic scale, a phenomenon "
    "the domain-adaptation literature has benchmarked in other contexts."))
A(para(
    "That literature has raised this concern before. Benchmark studies have shown that measured "
    "differences between generalization algorithms depend heavily on the model selection criterion "
    "[4], that unsupervised scorers on unlabeled target data are a viable alternative [5], [6], and "
    "that reported adaptation gains often shrink under standardized protocols [7]. Classical theory "
    "ties target error to a source-target divergence that source validation cannot observe [8]. Most "
    "proposed remedies for shift itself either require labeled data from the target hospital, which is "
    "precisely what is unavailable before deployment, or rely on having many training environments, "
    "which clinical settings rarely provide [9], [10]. Our contribution is not the idea but the "
    "evidence: real multi-hospital electronic health record data at full scale, where the shift is "
    "genuine rather than simulated."))
A(figure("rIdFig0", "Overview", 0.80))
A(para(
    "Structure of the study. PCL is the experimental vehicle, not the claim: its constraint weight "
    "provides the hyperparameter whose selection is at issue, and correcting the four evaluation "
    "confounds is what exposes the selection failure that the rest of the paper measures and fixes.",
    "figurecaption", keep=True))
A(para(
    "Fig. 1 summarizes that structure. Section II describes the PCL framework and its re-evaluation. Section III diagnoses source-domain "
    "validation and explains why it fails here. Section IV evaluates label-free alternatives that use "
    "unlabeled target data. Section V discusses scope, practical adoption, and future work."))

# ── II. PCL framework and rigorous evaluation ────────────────────────────────
A(para("PCL Framework and Rigorous Evaluation", "Heading1"))
A(para("Method and Objective", "Heading2"))
A(para(
    "PCL pretrains a six-layer transformer [11] (d = 256, 8 heads, about 3.2M parameters) on 48-hour, "
    "hourly-binned ICU windows using masked-value prediction [12]. The intuition is that the "
    "relationships physiology imposes among simultaneously measured variables hold at every hospital, "
    "even where devices, analyzers, and ordering habits do not, so a representation forced to respect "
    "them should depend less on site-specific measurement artifacts. The training objective adds a "
    "differentiable penalty for violating those relationships among the model's reconstructed outputs:"))
A(eq([("L", True, False), (" = ", False, False),
      ("L", True, False), ("mask", False, True),
      (" + ", False, False), ("\u03bb", True, False), (" ", False, False),
      ("L", True, False), ("phys", False, True)]))
A(para(
    "The first term is the masked-value reconstruction loss. The second is the mean penalty across "
    "three physiological relations: an absolute residual on the mean arterial pressure identity, and "
    "squared residuals on the Henderson-Hasselbalch equation and the Severinghaus oxygen dissociation "
    "relation. The penalty is applied only at timesteps where all required variables were observed and "
    "at least one was masked, so the model must impute from cross-variable structure rather than copy "
    "its input. The constraint weight \u03bb trades the two terms off, and it is the hyperparameter whose "
    "selection this paper studies: it is a natural choice for that role because, as Section III shows, "
    "it moves cross-hospital performance far more than it moves source validation performance."))

A(para("Study Design and Cohorts", "Heading2"))
A(para(
    "The separation between sites is strict: every training and selection decision is made on "
    "PhysioNet Site A, no target-site data is used during training whether labeled or unlabeled, and "
    "target labels are touched only when reporting final numbers. Table I summarizes the resulting "
    "cohorts. Sepsis prevalence differs substantially across systems, which is itself part of the "
    "shift being studied."))
A(para("Evaluation cohorts after filtering to adult stays of at least 24 hours.", "tablehead"))
A(table([
    ["Cohort", "Stays", "Sepsis+", "Role"],
    ["PhysioNet Site A", "10,381", "3.3%", "train / val"],
    ["PhysioNet Site B", "14,779", "2.1%", "target"],
    ["MIMIC-IV", "74,607", "30.9%", "target"],
    ["eICU-CRD", "130,446", "8.8%", "target"],
], [1500, 1000, 1100, 1300]))
A(para(
    "Models are trained on PhysioNet Challenge 2019 Site A [13] and evaluated zero-shot on three held-out "
    "cohorts: PhysioNet Site B (same collection protocol), MIMIC-IV [14] (74,607 stays), and eICU-CRD [15] "
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
    "Constraint availability is highly uneven, however, and this bounds what the violation score can "
    "be used for. The blood-pressure identity is computable at 70 to 80% of stay-hours, but the "
    "acid-base constraints at only 0.6 to 1.8%, because arterial blood-gas panels are drawn "
    "infrequently; on the loss actually optimized, the Henderson-Hasselbalch term contributes roughly "
    "one twenty-seventh the magnitude of the blood-pressure term. Because the score averages whichever "
    "constraints happen to be computable, and the constraints occupy very different magnitude ranges, "
    "its per-site mean partly tracks laboratory ordering practice rather than physiological "
    "consistency. Site A charts bicarbonate in 8.1% of stay-hours against 0.16% at Site B, a "
    "fifty-fold difference. On raw data with no model involved the blood-pressure residual is "
    "essentially identical at the two sites (0.0188 versus 0.0206), while the sparser Severinghaus "
    "residual is four times larger in scale and differs by a factor of 8.6, which reproduces the "
    "observed cross-site gap in violation with no model at all. The penalty remains a valid training "
    "signal, applied per timestep and never compared across sites, but its per-site average must not "
    "be read as a site-level shift statistic without standardizing each constraint separately."))

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
    "standard reference implementation [16], [17], yielding positive rates of 30.9% and 8.8% "
    "respectively; clinical and claims-based sepsis definitions are known to diverge substantially [18].",
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
    "With all four corrected, PCL no longer improved cross-hospital transfer. Table II reports the "
    "corrected comparison over three seeds. PCL is below ERM at every site, and a gradient-boosted "
    "tree [19] trained on simple per-stay summary statistics matches or exceeds both neural models, which "
    "further questions the architecture. A group-distributionally-robust baseline [10], once corrected for "
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
    "Expanding the input from nine physiological channels to seventeen improved external transfer for "
    "both methods and was retained throughout. Two unit-harmonization defects surfaced while doing so "
    "and would each have silently damaged transfer: MIMIC-IV charts temperature in Fahrenheit, and "
    "eICU-CRD records it in nurse charting rather than the periodic vitals table, leaving temperature "
    "observed for 82% of source hours but only 6% of target hours."))
A(para(
    "Two further observations support the negative result. Invariant risk minimization [9] with 30 "
    "hospital environments drawn from eICU performed no better than ERM (0.788 versus 0.792), "
    "indicating the failure is not confined to the low-environment regime. Separately, run-to-run "
    "variability across seeds reached 4 to 7 AUROC points, which exceeds most of the differences "
    "originally reported as method effects."))
A(para(
    "The negative result is what makes the rest of the paper possible. With no configuration "
    "meaningfully better than the others, the constraint weight becomes a clean test bed: target "
    "performance still differs sharply across the sweep, so whether a label-free criterion can tell "
    "the configurations apart is answerable without a method effect confounding the answer."))

# ── III. Diagnosing source-domain validation ─────────────────────────────────
A(para("Diagnosing Source-Domain Validation", "Heading1"))
A(para("The Validation-Target Disconnect", "Heading2"))
A(para(
    "Correcting the third confound raised the question that outlives PCL itself. If the constraint "
    "weight must be chosen without touching target data, does source-hospital validation choose it "
    "well? We swept the constraint weight over six values, recorded source validation AUROC and true "
    "AUROC at each held-out site, and compared the configuration that validation would select against "
    "the best configuration available."))
A(para(
    "Fig. 2 shows the disconnect. Across the whole sweep, source validation AUROC spans about one "
    "point (0.820 to 0.835) while target AUROC spans roughly eleven (0.548 to 0.750). Validation is "
    "nearly flat precisely where the target differences are largest, so it carries little information "
    "about which configuration will deploy well. In this seed source validation selects the "
    "unconstrained model, which is the weakest configuration at every external site, while the best "
    "available weight (0.5) is one validation would rank fifth of six."))
A(figure("rIdFig1", "ValidationGap", 0.86))
A(para(
    "The validation-target disconnect. Each point is one (constraint weight, target site) pair: "
    "horizontal position is that configuration’s source-hospital validation AUROC, vertical position "
    "its true zero-shot AUROC at that site. The cloud is almost vertical — validation varies by only "
    "0.015 across the sweep while target performance varies by 0.20 — and the weak trend that exists "
    "runs downward rather than upward. The annotated point is the configuration source validation "
    "selects, among the worst-performing at every external hospital; the best available configuration "
    "is marked for comparison.",
    "figurecaption", keep=True))
A(para(
    "We stress that the best-performing weight is not stable across seeds: in a second seed the "
    "unconstrained model was strongest at two of three sites. This instability is itself part of the "
    "finding. It means the practical question is not which weight is universally best, but whether "
    "any available selection criterion can identify a good one for a given deployment."))

A(para("Why Source Validation Fails Under Shift", "Heading2"))
A(para(
    "The mechanism is worth stating explicitly, because it predicts where else the failure should "
    "appear. Source validation data is drawn from the same hospital the model was fit to, so it "
    "contains the same measurement artifacts, in the same proportions, with the same predictive value. "
    "The configurations in our sweep differ mainly in how heavily they lean on those artifacts rather "
    "than on cross-variable physiological structure. A configuration that leans on them heavily is not "
    "penalized on source validation, because there the artifacts are still present and still "
    "predictive; it is penalized only at a hospital where the artifact is charted differently or not "
    "at all. Source validation is therefore blind along precisely the axis that separates the "
    "candidates, which is why it is nearly flat while target performance is not."))
A(para(
    "This also explains why the failure is not a variance problem that more validation data would "
    "solve. The source validation estimate is not noisy; it is measuring a quantity, in-distribution "
    "accuracy, that is close to constant across the sweep. Enlarging the validation split sharpens an "
    "estimate of the wrong quantity. What is needed is a criterion computed where the shift actually "
    "is, which means a criterion evaluated on target-hospital data."))

# ── IV. Unsupervised model selection ─────────────────────────────────────────
A(para("Empirical Study of Unsupervised Model Selection", "Heading1"))
A(para("Criteria and Metrics", "Heading2"))
A(para(
    "We therefore evaluated selection criteria that use unlabeled target data only. For each trained "
    "configuration and each target site, we computed four label-free scores in a single "
    "inference-only pass: masked-reconstruction error, physiological constraint violation, predictive "
    "entropy, and distance between source and target representation centroids. Each score is "
    "lower-is-better and is used to rank configurations."))
A(para(
    "We report two metrics. Spearman rank correlation measures whether a score orders configurations "
    "the way true target AUROC does; a more negative value is better. Selection regret is how far the "
    "configuration a criterion selects falls below the best available configuration in true target "
    "AUROC, so zero is optimal. Both are computed over three seeds and three target sites, "
    "and are reported in Table III."))
A(para("Label-free selection criteria. Rank correlation with true target AUROC "
       "(more negative is better) and mean selection regret in AUROC below the best available "
       "configuration (closer to zero is better), "
       "over 3 seeds and 3 sites.", "tablehead"))
A(table([
    ["Selection criterion", "Rank corr.", "Mean regret"],
    ["Reconstruction error", "-0.459", "0.012"],
    ["Constraint violation", "-0.426", "0.016"],
    ["Representation distance", "+0.002", "0.030"],
    ["Predictive entropy", "+0.150", "0.076"],
    ["Random selection", "\u2014", "0.043"],
    ["Source validation", "\u2014", "0.052"],
], [2500, 1200, 1200]))
A(para(
    "Per-seed behavior matters, since a criterion that only works on one seed is of no practical use. "
    "Reconstruction error is negative in all three (-0.600, -0.521, -0.255), as is constraint "
    "violation (-0.608, -0.391, -0.280), though both decay in strength; predictive entropy is "
    "positive in every seed (+0.216, +0.170, +0.065) and representation distance changes sign."))
A(para(
    "Three findings follow. First, source-hospital validation performs no better than selecting a "
    "configuration at random, and in two of three seeds it is worse; the difference from random is "
    "smaller than seed-to-seed variability, so we claim only that it is uninformative, not reliably "
    "harmful. Second, reconstruction error on unlabeled target data reduces mean regret from 0.052 to "
    "0.012, recovering most of the gap to oracle selection, and its advantage is consistent in sign "
    "across all three seeds. Third, predictive entropy and representation distance are not viable "
    "substitutes here; entropy is actively misleading, selecting the worst configuration in every "
    "seed."))

A(para("Why Reconstruction Error Works", "Heading2"))
A(para(
    "Reconstruction error succeeds for the reason source validation fails: it is computed where the "
    "shift is. Masked reconstruction asks the encoder to predict held-out measurements from the rest "
    "of the window, so it scores how well the cross-variable structure the model learned at the source "
    "hospital actually holds in target data. A configuration that absorbed source-specific "
    "measurement structure reconstructs target windows poorly, and the same encoder feeds the "
    "classifier, so the two degrade together. Crucially the score needs inputs only, so unlike target "
    "AUROC it is observable before deployment, and unlike source validation it is not blind to the "
    "shift. It is an observable proxy for a quantity the classical bounds treat as unobservable, "
    "namely the divergence between source and target [8]."))
A(para(
    "That framing also bounds the criterion. It measures how well the encoder fits the target input "
    "distribution, not how well the label-generating process transfers, so it cannot detect "
    "degradation coming purely from a change in how the outcome is defined, which is exactly what our "
    "first confound was."))
A(para(
    "Notably, the physiological constraint violation score does not outperform generic reconstruction "
    "error. The useful signal is that the model reconstructs target data poorly when it is configured "
    "badly, an observation consistent with denoising-autoencoder pretraining [20], and domain-specific "
    "physiological knowledge adds nothing on top of that. This is a negative result for our original "
    "method and a positive one for practitioners, since the effective criterion requires no domain "
    "expertise, no target labels, no additional parameters, and one inference pass."))

A(para("Practical Recommendation", "Heading2"))
A(para(
    "The procedure we would recommend is deliberately unglamorous. Train the candidate configurations "
    "as usual; before deployment, collect unlabeled data from the target hospital, run one forward "
    "pass per candidate to obtain its masked-reconstruction error on that data, and select the "
    "configuration with the lowest error. The cost is a single inference pass per candidate, no "
    "additional parameters, and no target labels, and in our experiments it recovered roughly three "
    "quarters of the gap between standard practice and oracle selection."))
A(para(
    "Adoption depends on when unlabeled target data is available, and the requirement is mild: input "
    "channels only, for a modest number of historical stays, with no outcome ascertainment, no chart "
    "review, and no labeling. Any hospital preparing to deploy a model already holds such data, and "
    "holds it at the point in the timeline where the selection decision is made."))
A(para(
    "Governance considerations are real but tractable. Patient-level data frequently cannot leave the "
    "institution, and the procedure does not require that it does: the computation is forward passes "
    "only, so candidate models can be shipped to the hospital and one scalar per candidate returned. "
    "What is still required is an agreement permitting on-site execution, and attention to the fact "
    "that a returned scalar remains a statistic of protected data. Where even that is not permitted "
    "before contracting, the criterion cannot be applied, and our results indicate the fallback "
    "should be treated as providing no information rather than a weak signal."))

# ── V. Discussion and future work ────────────────────────────────────────────
A(para("Discussion and Future Work", "Heading1"))
A(para(
    "Conventional source-domain validation is not reliable here: across our sweep it carried no more "
    "information about deployment performance than choosing at random, and was nearly flat precisely "
    "where target performance varied most. Scoring candidates by reconstruction error on unlabeled "
    "target data is cheap, needs no labels and no domain knowledge, and recovered most of the "
    "achievable performance. For teams deploying clinical models across sites, that substitution is a "
    "small change to standard practice with a measurable payoff."))
A(para(
    "The PCL study earns its place as the vehicle rather than the conclusion. Reporting its failure "
    "plainly matters: each confound is easy to introduce accidentally, and any one alone would have "
    "produced a publishable-looking gain. We would also encourage reporting seed variability as a "
    "matter of course. Run-to-run variation here reached 4 to 7 AUROC points, larger than most "
    "published cross-hospital improvements we are aware of, including the one we originally believed "
    "we had found. A single-seed comparison at this scale is not informative regardless of how large "
    "the reported difference is."))
A(para(
    "What we demonstrate is an effective empirical strategy, not a general solution, and its scope "
    "should be read accordingly. Our evaluation covers a single downstream task, sepsis onset "
    "prediction, and a single architecture; standard multi-task ICU benchmarks [21] and the wider "
    "sepsis-prediction literature [22] offer natural extensions. Whether the selection result "
    "holds for other clinical targets or model families is untested. Three seeds is a small sample "
    "given the 4 to 7 point run-to-run variance we measured, and secondary comparisons should be read "
    "with that in mind. Our external labels are re-derived Sepsis-3 rather than chart review, so "
    "residual label noise remains. The criterion itself is a proxy for input-distribution fit, so by "
    "construction it cannot see label-definition shift."))
A(para(
    "Three directions follow. Our proposed mechanism predicts the failure should be worst when "
    "candidate configurations differ in their reliance on site-specific measurement artifacts, which "
    "is testable on benchmarks where that reliance can be manipulated. Reconstruction error is one "
    "member of a family of encoder-side target statistics, and a systematic comparison at clinical "
    "scale would establish whether it is the right member. Finally, extension to shifts not driven by "
    "measurement practice, and to domains outside ICU time series, remains open."))

# ── Acknowledgment ───────────────────────────────────────────────────────────
A(para("Acknowledgment", "Heading5"))
A(para(
    "The authors thank the PhysioNet team and the MIMIC-IV and eICU-CRD Collaborative Research "
    "Database maintainers for curating and providing access to the credentialed datasets used here."))

# ── References ───────────────────────────────────────────────────────────────
# Ordered by first appearance in the text, per IEEE style. If a citation is
# moved, added, or removed, this list must be resequenced to match.
A(para("References", "Heading5"))
REFS = [
    # [1]-[3]: cross-hospital degradation, cited together in the opening paragraph.
    "S. G. Finlayson et al., “The clinician and dataset shift in artificial intelligence,” N. Engl. J. Med., vol. 385, no. 3, pp. 283–286, 2021.",
    "A. Wong et al., “External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients,” JAMA Intern. Med., vol. 181, no. 8, pp. 1065–1070, 2021.",
    "J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, “Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study,” PLoS Med., vol. 15, no. 11, e1002683, 2018.",
    # [4]-[8]: model selection under shift; [9]-[10] are the remedies that need
    # what clinical settings lack, and double as the IRM and Group DRO baselines.
    "I. Gulrajani and D. Lopez-Paz, “In search of lost domain generalization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.",
    "Y. Lalou, T. Gnassounou, A. Collas, A. de Mathelin, O. Kachaiev, A. Odonnat, A. Gramfort, T. Moreau, and R. Flamary, “SKADA-Bench: benchmarking unsupervised domain adaptation methods with realistic validation on diverse modalities,” arXiv:2407.11676, 2024.",
    "K. You, X. Wang, M. Long, and M. I. Jordan, “Towards accurate model selection in deep unsupervised domain adaptation,” in Proc. ICML, 2019, pp. 7124–7133.",
    "K. Musgrave, S. Belongie, and S.-N. Lim, “Unsupervised domain adaptation: a reality check,” arXiv:2111.15672, 2021.",
    "S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan, “A theory of learning from different domains,” Mach. Learn., vol. 79, no. 1–2, pp. 151–175, 2010.",
    "M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz, “Invariant risk minimization,” arXiv:1907.02893, 2019.",
    "S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, “Distributionally robust neural networks for group shifts,” in Proc. ICLR, 2020.",
    # [11]-[15]: architecture, pretraining objective, and the three data sources.
    "A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017.",
    "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL-HLT, 2019, pp. 4171–4186.",
    "M. A. Reyna et al., “Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019,” Crit. Care Med., vol. 48, no. 2, pp. 210–217, 2020.",
    "A. E. W. Johnson et al., “MIMIC-IV, a freely accessible electronic health record dataset,” Sci. Data, vol. 10, no. 1, 2023.",
    "T. J. Pollard et al., “The eICU Collaborative Research Database,” Sci. Data, vol. 5, 2018.",
    # [16]-[19]: label re-derivation and the tree baseline.
    "M. Singer et al., “The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3),” JAMA, vol. 315, no. 8, pp. 801–810, 2016.",
    "C. W. Seymour et al., “Assessment of clinical criteria for sepsis,” JAMA, vol. 315, no. 8, pp. 762–774, 2016.",
    "C. Rhee et al., “Incidence and trends of sepsis in US hospitals using clinical vs claims data, 2009–2014,” JAMA, vol. 318, no. 13, pp. 1241–1249, 2017.",
    "T. Chen and C. Guestrin, “XGBoost: a scalable tree boosting system,” in Proc. ACM SIGKDD, 2016, pp. 785–794.",
    # [20]: why reconstruction error carries signal.
    "P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Extracting and composing robust features with denoising autoencoders,” in Proc. Int. Conf. Mach. Learn. (ICML), 2008, pp. 1096–1103.",
    # [21]-[22]: extensions named in the discussion.
    "H. Harutyunyan, H. Khachatrian, D. C. Kale, G. Ver Steeg, and A. Galstyan, “Multitask learning and benchmarking with clinical time series data,” Sci. Data, vol. 6, no. 1, 96, 2019.",
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
