# SWEC Requirements Audit

Audit date: 2026-04-14

Primary manuscript checked:
- `/Users/hust-hwashin621m/Desktop/vietanhpaper-2/paper/main.tex`

Standalone submission bundle checked:
- `/Users/hust-hwashin621m/Desktop/vietanhpaper-2/paper/submission_full/main.tex`
- `/Users/hust-hwashin621m/Desktop/vietanhpaper-2/paper/submission_full/HIGHLIGHTS.md`
- `/Users/hust-hwashin621m/Desktop/vietanhpaper-2/paper/submission_full/COVER_LETTER.md`

Build verification:
- `submission_full` completed a full `bibtex + pdflatex + pdflatex` cycle successfully.
- Current compile status is clean enough for submission packaging; remaining warnings are minor overfull boxes, not missing files or broken references.

## Meets Journal Requirements

- Journal target is set correctly: `\journal{Swarm and Evolutionary Computation}` in `main.tex`.
- Editable LaTeX source exists and the standalone package includes `main.tex`, `references.bib`, `elsarticle.cls`, and `elsarticle-num.bst`.
- Title page is present with article title, author list, affiliations, and corresponding-author marker.
- Corresponding author is explicitly identified.
- Abstract is present and within the 250-word limit.
  - Measured abstract length: 200 words.
- Keywords are present and within the 1-7 keyword limit.
  - Current keyword count: 6.
- Core article structure is present.
  - Introduction
  - Related Work
  - Problem Formulation
  - Methods
  - Experimental Setup
  - Results
  - Discussion
  - Conclusion
- Competing-interests statement is present in the manuscript.
- Data and code availability section is present in the manuscript.
- Highlights file exists and meets the journal format guidance.
  - 5 bullets
  - each bullet <= 85 characters
- Cover letter exists and already states:
  - novelty and scope fit
  - not published elsewhere
  - not under consideration elsewhere
  - all authors approved
  - no competing interests
- All graphics referenced by the manuscript exist in the bundle.

## Missing Or Not Yet Satisfied

These are the main gaps that still need action before submission.

- CRediT author-contribution statement is missing.
  - SWEC says corresponding authors are required to acknowledge co-author contributions using CRediT roles.
  - No `Author Contributions`, `CRediT`, or equivalent section is present in `main.tex`.

- Funding statement is missing.
  - The guide says authors must disclose funding sources and recommend an explicit no-funding sentence when applicable.
  - No funding section or no-funding statement is present in `main.tex`.

- Required research-data repository compliance is not yet met clearly enough.
  - SWEC applies research data Option C: deposit research data in a relevant repository, cite and link the dataset in the article, or state why sharing is not possible.
  - The current manuscript only gives a GitHub repository link and an internal manifest path.
  - There is no formal dataset citation in the references and no explicit statement explaining why data cannot be deposited in a repository if that is the case.

- Submission-side declarations file is missing.
  - The journal says the declarations tool should always be completed and the resulting `.doc`/`.docx` file uploaded.
  - No declaration document is present in the paper workspace.

## Conditional Items You Must Confirm

- Declaration of generative AI use.
  - Required only if generative AI or AI-assisted tools were used in manuscript preparation beyond basic grammar/spell/reference tools.
  - There is currently no `Declaration of generative AI and AI-assisted technologies in the manuscript preparation process` section in `main.tex`.
  - If any AI drafting, rewriting, summarization, or literature-synthesis tools were used, add this section before the references.

- Acknowledgements.
  - Only needed if anyone provided research, writing, proofreading, or other help worth acknowledging.
  - No acknowledgements section is currently present.

- Full author email coverage.
  - The title page includes two email addresses, including the corresponding author.
  - The guide says author emails should be included if available.
  - This is not usually a hard blocker, but if emails for all authors are available, adding them would align better with the guide.

- Research-data archival scope.
  - If the repository already contains all data needed to reproduce the paper, the manuscript should say that explicitly and ideally provide a stable archival identifier.
  - If large checkpoints or outputs are excluded, state where they will be archived or why they cannot be shared.

## Optional Or Recommended Only

- Graphical abstract is encouraged, not required.
- Supplementary material is encouraged, not required.
- Video is optional.
- Software Impacts companion submission is optional.
- Open access choice is optional.

## Suggested Fix Order

1. Add a CRediT author-contribution section.
2. Add a funding section, or the no-funding sentence if applicable.
3. Tighten the data availability statement to satisfy Option C.
4. Add a generative-AI declaration if any such tools were used.
5. Prepare the separate declarations-tool `.doc` or `.docx` file for upload.

## Concrete Verdict

The manuscript is structurally close to submission-ready for SWEC, but it is not fully compliant yet.

Most likely blockers:
- missing CRediT statement
- missing funding statement
- incomplete research-data compliance
- missing declarations-tool upload file

Likely conditional blocker:
- missing generative-AI declaration if AI tools were used during manuscript preparation
