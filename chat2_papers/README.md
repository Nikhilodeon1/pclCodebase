# Chat 2 — Papers (URTC / Lin, and the withdrawn NeurIPS draft)

Self-contained. No datasets, no model code, no compute needed — every number is
already hardcoded in the build script.

## Contents
    paper/build_urtc.py                 builds the URTC .docx from the IEEE template
    paper/make_figure.py                generates the three figures
    paper/URTC_paper.docx               current build (5 pages)
    paper/conference-template-letter.docx   official IEEE letter template

The withdrawn original NeurIPS draft (`fmain_type2.tex`, `fmain.tex`) is NOT
here — it's quarantined at `_archive/pcl_original_draft/` (see warning below).
It used to sit in this folder and read as a third, active project; it isn't
one.

## Rebuild
    python paper/make_figure.py && python paper/build_urtc.py

Render and LOOK at it (Word/pandoc are not installed):
    "/c/Program Files/LibreOffice/program/soffice.exe" --headless --convert-to pdf --outdir . paper/URTC_paper.docx
    pdftoppm -jpeg -r 100 paper/URTC_paper.pdf page

## Warnings
* `/paper/` is gitignored in the parent repo — these files were NOT version
  controlled. Back up before large edits.
* `_archive/pcl_original_draft/fmain_type2.tex` still claims the PCL violation
  score works as an OOD detector (detection AUROC 0.70-0.79). That claim is
  now known to be confounded by the measurement-availability finding and was
  never re-audited. Do not reuse it without correcting. The URTC paper does
  not make this claim.
* Any `results/*.json` you find elsewhere in the parent repo is STALE single-seed
  demo output, not the paper's numbers. The verified full-scale numbers live in
  the handoff prompt and in build_urtc.py.

## IEEE template gotchas (each cost real debugging time)
* Styles auto-number: never type "Table I." / "Fig. 1." / "[1]" yourself.
  `tablehead`, `figurecaption` and `references` insert them.
* `Heading1` auto-numbers with Roman numerals; `Heading5` is the UNNUMBERED
  heading used for Acknowledgment and References.
* Column width is 5047 DXA. Wider tables overflow the column.
* `BodyText` fixes line height (w:line=11.40pt) which CLIPS inline images.
  Image paragraphs must use no body style plus w:lineRule="auto".
* Set image EMU from the actual pixel aspect ratio, or figures render squashed.
* Do not re-declare XML namespaces the template already has (duplicate
  attribute = file Word refuses to open).
