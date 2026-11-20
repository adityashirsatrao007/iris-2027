# IRIS 2027 submission audit

**Guideline source checked:** `C:\Users\aditya\Downloads\iris\IRIS 2027 – Author Guidelines for Final Paper Preparation and Submission.pdf`  
**Guideline length:** 8 pages  
**Internal deadline stated in the guideline:** 25 November 2026

## Executive result

The research content is substantially complete. After the format correction, the
paper source and PDF are now aligned with the two main format requirements. The
remaining blockers are the signed/legal submission materials.

The final LaTeX package now uses `svmult`, contains the required supporting
files, and compiles to exactly 10 pages. The humanized DOCX remains available as
an editable working copy, but the official submission package uses the matching
LaTeX source and PDF so the source/PDF pair is authoritative.

## Requirement-by-requirement status

| Requirement | Status | Action |
|---|---|---|
| Exactly 10 pages | PASS | `paper/main.pdf` and submission PDF are 10 pages |
| Official electrical-engineering proceedings template | PASS | Submission uses Springer `svmult`, not `llncs` |
| American English | PASS with editorial review | Perform final language review |
| No page numbers/running heads | PASS | `\\titlerunning` removed from submission source |
| Full author name | PASS | Verify spelling against agreement |
| City and country in affiliation | PASS | Current affiliation says Solapur, India |
| Corresponding-author email | PASS | `adityashirsatrao007@gmail.com` is present |
| Keywords | PASS | Five keywords are present |
| Abstract without footnotes | PASS | No abstract footnote detected |
| Heading capitalization | REVIEW | Some headings need title-case adjustment under the guideline |
| Figure captions below figures | PASS in LaTeX | Recheck after conversion to the official template |
| Table captions above tables | PASS in LaTeX | Recheck after conversion to the official template |
| Editable Word equations | **PENDING** | DOCX conversion must be checked; equations should be Word Math/MathType objects |
| Numeric square-bracket citations | PASS | `svmult`/`spmpsci` produces numeric square-bracket citations |
| DOI inclusion | REVIEW | Add DOI values where available |
| Final source files and PDF match | PASS | Submission PDF is compiled from submitted `main.tex` |
| Signed Contributor Agreement | **PENDING** | Complete and sign the supplied form |
| Permissions checklist | **MISSING** | Obtain/complete the Springer checklist or document why no permission is required |
| Third-party permissions | REVIEW | Document COCO, GenImage, and model/checkpoint licenses |

## Files found in the IRIS folder

The folder contains:

- `IRIS 2027 – Author Guidelines for Final Paper Preparation and Submission.pdf`
- `IRIS-2027-guidelines.md`
- `Contributor Agreement.docx`
- `Contributor-Agreement.md`
- `Research-Permission-Template-EN.docx`
- `Research-Permission-Template-EN.md`
- `Research-Release-Template-EN.docx`
- `Research-Release-Template-EN.md`
- `Turnitout_Humanized_Docs_20Nov2026_165140.zip`

No completed or signed agreement is present. No separate Permissions Checklist
file was found.

## Current paper files

- LaTeX source: `paper/main.tex`
- LaTeX PDF: `paper/main.pdf` — **10 pages**
- Editable DOCX: `paper/iris-27-paper.docx`
- Submission ZIP: `dist/iris-2027-final-submission.zip`

The DOCX is editable and contains the humanized wording, but it is not the
official Springer Word proceedings template. The final package uses the LaTeX
source as its authoritative source and the PDF has been compiled from that exact
source; the DOCX is retained separately as a working copy.

## Required final workflow

1. Complete the author and legal submission materials.
2. Verify all numbers against `docs/RESULTS.md` and
   `results/publication_metrics.json`.
3. Complete and sign the Contributor Agreement.
4. Complete the permissions checklist and attach any required permissions or
   license records.
5. Submit the source, matching PDF, signed agreement, permissions material,
   corresponding-author email, and any supplementary material as one package.

## Important conclusion

The paper source/PDF format is now aligned with the supplied template and
10-page requirement. Final submission remains conditional on completing the
signed Contributor Agreement and the permissions checklist/licence records.
