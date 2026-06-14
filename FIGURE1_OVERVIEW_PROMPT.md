# Figure 1 — Overview / Graphical-Abstract prompt

> Purpose: a single, polished **system-overview** figure for the paper's Figure 1 (and reusable as the Elsevier *graphical abstract*). It must reflect the paper's **diagnosis** framing — *every learned model falls below a popularity baseline under realistic evaluation* — NOT a "GAT is our method" framing.
>
> **Why the old `files/Figure_1.png` must be redrawn (it is now inaccurate):** it labels company nodes "Learnable d=64" (for the GNNs they are **fixed-random**; only LightGCN/NGCF learn an ID embedding), shows "GAT Encoder (2 layers, **4 heads**)" (our GAT is **single-head**, scatter-based), centers a single model (GAT) as "the method" (the paper evaluates **many** models and finds they **all fail**), and lists "dynamic uniform negative sampling" (evaluation uses **same-IPC hard negatives**). Keep the old figure's clean left-to-right flow style, but fix the content and add the central finding.
>
> Our accurate-but-plain Matplotlib version is `paper_figures/fig0_pipeline.png` (use it as the structural reference / fallback). The prompt below is for producing a more refined version with a design tool (draw.io / Figma / Illustrator) or a diagram-capable image model. **Because the figure contains exact technical text, prefer a vector design tool over a text-to-image model** (image models mangle small text); if using an image model, treat the text below as labels to be typeset cleanly, not paraphrased.

---

## One-line prompt (for a diagram/image tool)

> "A clean, professional, three-stage horizontal schematic for an academic IR/ML paper (Elsevier *Information Processing & Management* style; flat design, muted blue/red/grey/amber palette, white background, sans-serif, no drop shadows, color-blind-safe). Stage 1 'Heterogeneous graph': two blue circular **Patent** nodes (label 'frozen SBERT, 384-d') connected by a grey 'cites' arrow, each linked by amber 'transfer' arrows to three red rounded-square **Company** nodes (label 'fixed-random 64-d feature; ID embedding only for LightGCN/NGCF'); caption '370,666 patents · 122,519 companies · ~910k citations'. Stage 2 'Realistic evaluation protocol' as a 5-box left-to-right pipeline: (1) Temporal 70/15/15 split by registration date → (2) encode on **train-split edges only** → (3) dot-product score s(p,c)=hp·hc → (4) rank the true company against **100 same-IPC hard negatives** → (5) **average-rank** tie-break → NDCG@10 / AUC; small red flag under stage 2 reading '91.7% of test patents are unseen (cold-start)'. Stage 3 'Diagnosis' as a small horizontal bar chart: a tall grey bar 'MostPop 0.197' with a dashed 'popularity baseline' line, and shorter coloured bars for GAT, GraphSAGE, NGCF, SVD, MLP all **below** the line, titled 'every learned model falls below popularity; AUC ≈ chance'; two small badges 'Popularity bias (ρ up to 1.0)' and 'Extreme cold-start (91.7%)', and a dashed callout 'a random split would reverse this (SVD → 0.44)'. Thin grey arrows connect the three stages left to right."

---

## Detailed specification (hand to a designer)

**Canvas.** Landscape, ~2:1 (e.g. 1800×900 px or A-width vector). White background. Palette: patents `#4477AA` (blue), companies `#EE6677` (red), citation/structure `#888888` (grey), transfer/protocol accents `#CC8844`/`#CCBB44` (amber), encoder `#66AA66` (green). Sans-serif (Helvetica/Arial/Inter). Color-blind-safe; do not rely on red/green alone.

**Three panels, left → right, joined by thin grey arrows; a slim title strip above each.**

**Panel A — "Heterogeneous graph" (the data).**
- Two blue **circles** labelled **P** stacked vertically; a grey arrow between them labelled *cites*.
- Three red **rounded squares** labelled **C** on the right; faint amber arrows from each P to the Cs labelled *transfer* (the prediction target).
- Small labels: top-left "Patent node — frozen SBERT, 384-d"; top-right "Company node — fixed-random 64-d (GNN); learnable ID embedding only for LightGCN/NGCF".
- Footer line: "Heterogeneous graph G=(V,E): patents ∪ companies; edges = citations ∪ realized transfers. 370,666 patents · 122,519 companies · ≈910k citation edges."

**Panel B — "Realistic evaluation protocol" (the contribution).**
- Five outlined boxes in a row, each a distinct accent color, joined by `→`:
  1. **Temporal split** — "70/15/15 by transfer registration date (forward in time)"
  2. **Encode** — "heterogeneous GNN on train-split edges only (no test leakage)"
  3. **Score** — "s(p,c) = hₚ·h_c (dot product)"
  4. **Hard negatives** — "rank the true company vs **100 same-IPC** companies"
  5. **Tie-break** — "**average rank** → NDCG@10, AUC"
- A red ribbon/flag beneath: **"91.7% of test patents are unseen in training (extreme cold-start)."**

**Panel C — "Diagnosis: the central finding".**
- A compact horizontal bar chart of NDCG@10: dashed vertical line at **0.197** labelled *"popularity skyline (MostPop)"*; bars for MostPop-IPC 0.199, NGCF 0.196, then **all learned below the line**: MLP 0.150, GraphSAGE+logQ 0.125, GAT+logQ 0.107, GraphSAGE 0.084, GAT 0.074, SVD 0.037. Title: *"Every learned model falls below the popularity baseline; AUC ≈ chance (0.42–0.60)."*
- Two pill-shaped badges: **"Popularity bias — score↔popularity ρ up to 1.0; 56% hard-neg inversions"** and **"Cold-start — 91.7% unseen patents; SVD = 0.000 on new patents."**
- A dashed callout box: **"A conventional random split reverses the verdict: SVD jumps to 0.442 (AUC 0.83), the best model — §RQ1."**

**Tone.** Diagnostic and neutral; no model is the "hero." The visual story is: *realistic protocol (B) applied to a standard graph setup (A) → learned models lose to popularity (C).*

---

## Caption to use in the manuscript (Figure 1)

> **Figure 1.** Overview of the task and the central finding. We frame patent technology-transfer demand prediction as link prediction on a heterogeneous patent–citation–transfer graph (A) and evaluate it under a leakage-free protocol — a temporal 70/15/15 split, encoding on training-split edges only, ranking the true acquirer against 100 same-IPC hard negatives, and an average-rank tie-break (B). Under this protocol every learned model (GraphSAGE, GAT, LightGCN, NGCF, SVD, a text MLP) falls below a simple most-popular baseline and discriminates near chance (C), which we trace to popularity bias and an extreme (91.7%) patent cold-start regime; a conventional random split would have reversed this verdict.
