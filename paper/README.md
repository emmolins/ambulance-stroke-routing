# T-ASE manuscript

LaTeX source for the IEEE Transactions on Automation Science and Engineering submission, built from the JAMA Network Open manuscript (rejected; reviewer reports in the project notes).

## Build

```
cd paper
latexmk -pdf main.tex
```

Needs a TeX distribution with `IEEEtran.cls` (bundled here, v1.8 2012 variant; replace with the official `IEEEtran.cls` v1.8b from the IEEE template package before submission).

## Layout

- `main.tex`: preamble, author block, page budget, drafting macros
- `sections/00_abstract.tex` to `08_conclusion.tex`: one file per section, each headed by a comment giving its page budget and its source in the JAMA manuscript
- `refs.bib`: JAMA references plus new ones; entries marked VERIFY need checking
- `figures/`: JAMA figures as placeholders (`jama_*.png`); regenerate from the reruns

## Drafting markers

- Blue text (`\new{}`): to be written; not in the JAMA manuscript
- Red `[PENDING: ...]` (`\pending{}`): a number or figure awaiting the reruns
- Grey small text (`\jama{}`): where the text came from and what changed

Set `\draftfalse` in `main.tex` to hide all markers; set `\anontrue` for the double-anonymous submission (removes authors and acknowledgments).

## Resolved: outcome curve sign and outcome definition

Checked against Holodinsky et al. 2018 (Table, main text) on 2 Oct 2026: the EVT curve is `0.3394 + 0.00000004 t^2 - 0.0002 t` (plus sign), as the code has it; the JAMA supplement eMethods 1 carried a sign typo. The LVO combination `P_IVT + (1 - P_IVT) P_EVT` matches Holodinsky eMethods section C. The outcome is "excellent outcome", mRS 0-1 at 90 days (not 0-2); the paper now says so everywhere.

## T-ASE constraints

10 pages including references (overlength charge beyond 10, hard cap 12); double-anonymous; Note to Practitioners 100 to 300 words immediately after the abstract; abstract 200 words; Regular Paper; one revise-and-resubmit maximum.
