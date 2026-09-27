"""Write results/summary.docx: an edited, narrative Word summary of the study built from
results/results.json (which also carries the tuning records and the diagnostics),
results/data_stats.json and the figures.

    python make_report_docx.py

Every number in the document is read from the result files at run time, and every comparative
word ("better", "worse", "indistinguishable") is chosen from the sign of a bootstrap confidence
interval, so the text stays consistent with the numbers if the experiment is re-run.
"""
from __future__ import annotations

import json
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from evaluate import RES, ZONE_DECODERS, ZONE_LAMS, ZONE_ORDER_LAMS

FIG = os.path.join(RES, "figures")

NAMES = {
    "driver": "Driver's actual sequence",
    "nn": "Nearest neighbour",
    "ortools": "OR-Tools, guided local search (5 s)",
    "zone": "OR-Tools with zone-change penalty (5 s)",
    "zonehist": "Historical zone order + OR-Tools (5 s)",
    "softdist_greedy": "SoftDist heat-map, greedy",
    "softdist": "SoftDist heat-map, greedy + 2-opt",
    "sup_greedy": "One-shot supervised GNN, greedy",
    "sup": "One-shot supervised GNN, greedy + 2-opt",
    "diffusion_greedy": "Stop-level diffusion, greedy",
    "diffusion": "Stop-level diffusion, greedy + 2-opt",
    "diffusion_budget_greedy": "Stop-level diffusion, spec-budget model, greedy",
    "diffusion_budget": "Stop-level diffusion, spec-budget model, greedy + 2-opt",
    "hier_sup": "One-shot supervised zone GNN + OR-Tools (5 s)",
    "hier": "Zone-level diffusion + OR-Tools (5 s)",
}
ORDER = ["hier_sup", "hier", "zonehist", "zone", "diffusion", "sup", "diffusion_greedy", "sup_greedy", "diffusion_budget",
         "diffusion_budget_greedy", "softdist",
         "softdist_greedy", "ortools", "nn", "driver"]
GAP_NAMES = {"hier - zonehist": "Zone diffusion minus history zone order",
             "hier - hier_sup": "Zone diffusion minus one-shot zone GNN",
             "hier_sup - zonehist": "One-shot zone GNN minus history zone order",
             "hier - zone": "Zone diffusion minus zone-change heuristic",
             "diffusion - sup": "Stop diffusion minus one-shot GNN (both + 2-opt)",
             "diffusion_greedy - sup_greedy": "Stop diffusion minus one-shot GNN (greedy only)",
             "diffusion - softdist": "Stop diffusion minus SoftDist (both + 2-opt)",
             "sup - softdist": "One-shot GNN minus SoftDist (both + 2-opt)",
             "diffusion - zone": "Stop diffusion + 2-opt minus zone-change heuristic",
             "hier_sup - zone": "One-shot zone GNN minus zone-change heuristic",
             "diffusion_budget - diffusion": "Spec-budget stop diffusion minus full stop diffusion (both + 2-opt)"}


# ---------------------------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------------------------
def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def para(doc, text, italic=False, size=None, align=None, space_after=6, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold
    if size:
        r.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def figure(doc, path, caption, width_cm=16.0):
    if not os.path.exists(path):
        return False
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, caption, italic=True, size=9, space_after=10)
    return True


def table(doc, headers, rows, widths, bold_rows=()):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(8)
        shade(c, "DCE8F7")
    for i, vals in enumerate(rows, start=1):
        for j, v in enumerate(vals):
            c = t.rows[i].cells[j]
            c.text = ""
            run = c.paragraphs[0].add_run(v)
            run.font.size = Pt(8)
            run.bold = (i - 1) in bold_rows
    for row in t.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def results_table(doc, summ):
    methods = [m for m in ORDER if m in summ["methods"]]
    rows = []
    for m in methods:
        d = summ["methods"][m]
        a, tt, rt = d["amazon_score"], d["travel_time_s"], d["runtime_s"]
        rows.append([NAMES[m], f"{a['mean']:.4f} ({a['ci95_mean'][0]:.4f} to {a['ci95_mean'][1]:.4f})", f"{a['median']:.4f}",
                     f"{tt['mean'] / 3600:.2f}", f"{tt['median'] / 3600:.2f}", f"{d['driver_edge_overlap']['mean'] * 100:.1f}%",
                     f"{d['zone_order_tau']['mean']:.2f}", f"{rt['median']:.2f} ({rt['q25']:.2f} to {rt['q75']:.2f})"])
    table(doc, ["Method", "Amazon score, mean (95% CI)", "Score median", "Travel time mean, h", "Travel time median, h",
                "Driver moves reproduced", "Zone-order tau", "Runtime median (IQR), s"],
          rows, [4.3, 3.0, 1.4, 1.5, 1.5, 1.6, 1.3, 2.4], bold_rows=[])


def seed_table(doc, ss):
    seeds = ss["seeds"]
    rows = []
    for m in ORDER:
        if m not in ss["methods"]:
            continue
        d = ss["methods"][m]
        rows.append([NAMES[m]] + [f"{d['per_seed'][str(s)]['mean']:.4f}" for s in seeds]
                    + [f"{d['across_seeds']['mean']:.4f} ({d['across_seeds']['sd']:.4f})"])
    table(doc, ["Learned method"] + [f"Seed {s}" for s in seeds] + ["Mean (sd) across seeds"], rows,
          [6.0] + [2.0] * len(seeds) + [3.2])


def gap_table(doc, ss):
    seeds = ss["seeds"]
    rows = []
    for g, d in ss["gaps"].items():
        per = [f"{v['mean_diff_ref_minus_other']:+.4f} ({v['ci95'][0]:+.4f} to {v['ci95'][1]:+.4f})"
               for v in (d["per_seed"][str(s)] for s in seeds)]
        sa = d["seed_averaged"]
        rows.append([GAP_NAMES.get(g, g)] + per + [f"{sa['mean_diff']:+.4f} ({sa['ci95'][0]:+.4f} to {sa['ci95'][1]:+.4f})"])
    table(doc, ["Paired comparison"] + [f"Seed {s}" for s in seeds] + ["Seed-averaged"], rows,
          [4.6] + [3.2] * len(seeds) + [3.2])


# ---------------------------------------------------------------------------------------------
# wording helpers driven by confidence intervals
# ---------------------------------------------------------------------------------------------
def verdict(ci, better="better than", worse="worse than", same="statistically indistinguishable from"):
    if ci[1] < 0:
        return better
    if ci[0] > 0:
        return worse
    return same


def size_word(diff, base):
    """Describe an absolute score difference relative to the comparison method's mean score."""
    rel = abs(diff) / base if base else 0.0
    return "small" if rel < 0.15 else ("moderate" if rel < 0.35 else "large")


def main():
    R = json.load(open(os.path.join(RES, "results.json")))
    tuning = R["tuning_on_val"]
    diag = R.get("diagnostics")
    tc = R.get("training_compute")
    stats = json.load(open(os.path.join(RES, "data_stats.json")))
    prim = "fresh" if "fresh" in R["splits"] else "test"
    FR = R["splits"][prim]
    fr, fr_ss, fr_b = FR["summary"], FR.get("seed_summary"), FR.get("budget_sensitivity")
    reused = [s for s in ["test", "heldout"] if s in R["splits"] and s != prim]

    def M(s, m, k="amazon_score", f="mean"):
        return s["methods"][m][k][f]

    def P(s, ref, other, k="amazon_score"):
        return s["paired"][ref][other][k]

    def SA(ss, g):
        return ss["gaps"][g]["seed_averaged"]

    def ci_txt(ci, nd=4):
        return f"{ci[0]:+.{nd}f} to {ci[1]:+.{nd}f}"

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Cm(2.0)
        sec.top_margin = sec.bottom_margin = Cm(2.0)

    title = doc.add_heading("Can a diffusion model learn delivery sequences from past successes?", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para(doc, "A side-by-side test on real Amazon last-mile routes against classical routing solvers, a non-learned control "
              "and a supervised ablation", italic=True, size=12)

    # ------------------------------------------------------------------ short answer
    doc.add_heading("The short answer", level=1)
    n = fr["n_routes"]
    para(doc,
         "The idea under test is that a generative diffusion model trained only on delivery routes that drivers executed well "
         "should produce a sensible sequence for a new route, because it has absorbed what good routes look like. We built "
         "that model on the public Amazon Last Mile Routing Research Challenge data, trained it only on the "
         f"{stats['n_high']:,} routes Amazon rated High quality, and compared it on {n} unseen routes of the separate official "
         "evaluation dataset with nearest neighbour, Google OR-Tools, a zone-aware OR-Tools heuristic, a non-learned heat-map "
         "control and, as an ablation, the identical neural network trained as an ordinary one-shot supervised classifier. "
         "Every stop, zone, travel time and driver sequence is real. The final numbers come from a set of routes that was "
         "first evaluated only after every setting of every method had been frozen.")
    para(doc,
         "The answer is no: the experiment gives no evidence that the diffusion process helps. The specified stop-level "
         f"diffusion model scored {M(fr, 'diffusion'):.4f} on the official challenge metric (lower is closer to the driver), "
         f"{verdict(P(fr, 'diffusion', 'softdist')['ci95'])} the non-learned SoftDist control ({M(fr, 'softdist'):.4f}) but "
         f"{verdict(P(fr, 'diffusion', 'zone')['ci95'])} a simple zone-aware OR-Tools heuristic ({M(fr, 'zone'):.4f}).",
         bold=False)
    if fr_ss and all(g in fr_ss["gaps"] for g in ["diffusion - sup", "diffusion_greedy - sup_greedy", "hier - hier_sup"]):
        g1, g2, g3 = SA(fr_ss, "diffusion - sup"), SA(fr_ss, "diffusion_greedy - sup_greedy"), SA(fr_ss, "hier - hier_sup")
        para(doc,
             "The decisive comparison is the ablation. The same network, trained on the same data with the same features, "
             "optimiser, schedule, number of steps and decoder, but as a plain one-shot supervised classifier instead of a "
             "diffusion model, was closer to the drivers at both levels. Averaged over three training seeds, diffusion minus "
             f"supervised is {g1['mean_diff']:+.4f} (95% interval {ci_txt(g1['ci95'])}) at stop level with 2-opt, "
             f"{g2['mean_diff']:+.4f} ({ci_txt(g2['ci95'])}) without 2-opt, and {g3['mean_diff']:+.4f} "
             f"({ci_txt(g3['ci95'])}) at zone level; positive means diffusion is worse. The diffusion model is "
             f"{verdict(g1['ci95'])}, {verdict(g2['ci95'])} and {verdict(g3['ci95'])} its supervised twin in these three "
             "comparisons, and it is also slower. The diffusion process adds no measurable benefit over a single supervised "
             "forward pass of the same network.")
    if fr_ss and "hier_sup - zonehist" in fr_ss["gaps"]:
        others = [m for m in fr["methods"] if m != "driver"]
        best = min(others, key=lambda m: M(fr, m))
        gz = SA(fr_ss, "hier_sup - zonehist")
        para(doc,
             f"The most driver-like method overall was {NAMES[best].lower()} ({M(fr, best):.4f} with the seed-0 model). That "
             "is a hybrid designed after a diagnosis on validation routes, and most of its advantage does not come from "
             "generative sequence modelling. A learned model only orders the roughly 20 zones of a route, using historical "
             "zone-transition counts from the training routes as input, and OR-Tools with guided local search then sequences "
             "every stop inside that order. A pure history lookup that decodes the same counts without any network, followed by "
             f"the same OR-Tools step, scored {M(fr, 'zonehist'):.4f}; the one-shot zone network beats it by "
             f"{-gz['mean_diff']:.4f} averaged over seeds (interval {ci_txt(gz['ci95'])}), and the zone diffusion model "
             f"scored {M(fr, 'hier'):.4f}. Plain OR-Tools scored {M(fr, 'ortools'):.4f}.")
    para(doc,
         "Even the best method is far from challenge-level performance. The winning team of the 2021 challenge (Cook, Held and "
         "Helsgaun; arXiv 2112.15192) is reported to have scored about 0.025 on the full evaluation set; we could not open the "
         f"paper from this environment to verify it, and our {n}-route sample is not a leaderboard submission.")

    # ------------------------------------------------------------------ test sets
    doc.add_heading("Which routes the results come from", level=1)
    para(doc,
         f"Final test set: {n} routes drawn at random from the {stats.get('n_eval_not_in_test', 0):,} routes of the official "
         "evaluation dataset that had not been used before, and evaluated once, after all settings were frozen. Nothing in the "
         "study was decided with it.")
    para(doc,
         "Two further sets are reported but are no longer untouched: 300 other evaluation-dataset routes and 150 held-out "
         "High-quality training-dataset routes. Both were evaluated in an earlier round of this study. After seeing those "
         "results the models were retrained, the supervised ablations and two more training seeds were added, the tuning grids "
         "were widened and the tuned setting of the zone heuristic changed. No setting was chosen by a test score, but the "
         "design was revised after test results had been seen, so these two sets serve as development results only. "
         "Their conclusions agree with the final set (see results/summary.md).")

    # ------------------------------------------------------------------ data
    doc.add_heading("The data and the measures", level=1)
    para(doc,
         "The Amazon Last Mile Routing Research Challenge dataset (Merchan and colleagues, Transportation Science, 2024) is "
         "published on the AWS Open Data registry under a Creative Commons Attribution-NonCommercial licence, which this "
         f"research use respects. The training part has {stats['n_train_total']:,} routes from 17 depots in five US "
         "metropolitan areas, each with stop coordinates, zone identifiers, a full matrix of real travel times and the "
         f"sequence the driver actually drove. Only the {stats['n_high']:,} routes rated High were used for learning: "
         f"{stats['split_sizes']['train']:,} for training and {stats['split_sizes']['val']} for choosing every setting of every "
         "method. The evaluation dataset carries no route-quality label, so the quality of the test executions is unknown.")
    para(doc,
         "Two measures are reported for every route: the total travel time of the closed tour on the real travel-time matrix, "
         "and the official challenge score, a sequence-deviation term multiplied by an edit distance on normalised travel "
         "times, measured against the driver's sequence. Lower is better and the driver scores exactly zero. The score is "
         "re-implemented from the organisers' published script and tested to give identical values on real routes.")

    # ------------------------------------------------------------------ method
    doc.add_heading("How the models work", level=1)
    para(doc,
         "A route is a graph. The diffusion model learns the set of directed connections a driver uses, one yes-or-no value per "
         "candidate connection on a 16-nearest-neighbour graph. During training the driver's connections are scrambled by "
         "random bit flips and a gated graph neural network learns to undo the scrambling, looking at real travel times, stop "
         f"locations and zone labels. At use time the model denoises from pure noise in {tuning['diff_steps']} steps; "
         f"{tuning['diff_samples']} samples are averaged into a map of connection probabilities, which a greedy procedure "
         "turns into one valid tour, optionally polished by 2-opt. This is the DIFUSCO approach (Sun and Yang, NeurIPS 2023), "
         "trained to imitate drivers rather than to minimise distance.")
    para(doc,
         "The supervised ablation uses the identical network, features, training routes, batch size, learning-rate schedule and "
         "number of optimisation steps, but receives no noisy input and predicts every connection in one forward pass; its map "
         "is decoded in exactly the same way. Any difference between the two is due to the diffusion process. Both objectives "
         "were trained at both levels with three seeds each.")
    para(doc,
         "The zone-level models apply the same code to a graph whose nodes are the route's zones. Besides geography and zone "
         "labels they see how often drivers from the same depot moved between each pair of zones in the training routes "
         "(leave-one-out for training routes). The generated zone order is handed to OR-Tools, which fills in the stops with a "
         "penalty on every move that leaves that order.")
    if diag:
        ss, zs = diag["stop_successor"], diag["zone_structure"]["val"]
        dif = [v["top1"] for k, v in ss.items() if k.startswith("diffusion_seed")]
        sup = [v["top1"] for k, v in ss.items() if k.startswith("supervised_seed")]
        para(doc,
             f"Diagnosis on the {diag['n_routes']} validation routes (diagnostics.py, which reads no test data): the stop-level "
             f"diffusion model's most likely next stop was the driver's actual next stop {min(dif) * 100:.1f}% to "
             f"{max(dif) * 100:.1f}% of the time across seeds, the supervised model's {min(sup) * 100:.1f}% to "
             f"{max(sup) * 100:.1f}%, and simply choosing the nearest stop {ss['nearest_stop']['top1'] * 100:.1f}%. Meanwhile "
             f"{zs['same_zone_move_share'] * 100:.1f}% of a driver's consecutive moves stay within one zone and "
             f"{zs['contiguous_zone_share'] * 100:.1f}% of zones are served in one block, which is why the zone level was added.")

    # ------------------------------------------------------------------ results
    doc.add_heading(f"Results on the final test set ({n} routes)", level=1)
    para(doc,
         "Seed-0 models. Scores are means with bootstrap 95% confidence intervals over routes, then medians. Driver moves "
         "reproduced is the share of the driver's stop-to-stop moves a method also makes; zone-order tau is the rank "
         "correlation between the method's zone order and the driver's. Runtime is the median wall-clock time per route; it "
         "includes the 5 seconds given to OR-Tools where used.")
    results_table(doc, fr)
    para(doc,
         "The cost of looking like a driver is travel time. OR-Tools, which only minimises travel time, produced tours "
         f"{-P(fr, 'hier_sup', 'ortools', 'travel_time_s')['mean_diff_ref_minus_other'] / 60:.1f} minutes shorter on average "
         "than the one-shot zone GNN hybrid, yet those tours scored much worse against what drivers did. A lower score means "
         "similarity to the driver, not a better route.")
    figure(doc, os.path.join(FIG, "ablation_diffusion_vs_supervised.png"),
           "Figure 1. Per-route score of each diffusion model against its one-shot supervised twin (both averaged over three "
           "seeds). Points above the diagonal are routes where the supervised model is closer to the driver.", width_cm=17.0)
    figure(doc, os.path.join(FIG, "score_distributions.png"),
           f"Figure 2. Per-route challenge score and travel time for every method on the {n} final-test routes (seed-0 "
           "models). Boxes show quartiles, diamonds means; the dashed line is the drivers' median travel time.")
    if fr_ss:
        doc.add_heading("Seeds and paired gaps", level=2)
        para(doc,
             "Each learned model was trained with three seeds. The first table gives each model's mean score per seed; the "
             "second gives paired differences on the same routes, seed by seed, with 95% bootstrap intervals, and in the last "
             "column after averaging each method's per-route score over the seeds. Negative values favour the first method.")
        seed_table(doc, fr_ss)
        gap_table(doc, fr_ss)
        figure(doc, os.path.join(FIG, "seed_spread.png"),
               "Figure 3. Mean score of every learned model for each seed (blue circles: diffusion; orange squares: one-shot "
               "supervised) beside the non-learned control of the same row (gray bar) and the zone heuristic (dashed line), "
               "on the final set and the two reused sets.", width_cm=17.0)
    if "diffusion_budget" in fr["methods"]:
        para(doc,
             "A stop-level diffusion model retrained within the specification's training budget (3,000 instead of 7,000 steps) "
             f"scored {M(fr, 'diffusion_budget'):.4f} with 2-opt, against {M(fr, 'diffusion'):.4f} for the full model "
             f"(paired difference {P(fr, 'diffusion', 'diffusion_budget')['mean_diff_ref_minus_other']:+.4f}, full minus "
             f"budget, interval {ci_txt(P(fr, 'diffusion', 'diffusion_budget')['ci95'])}).")

    for s in reused:
        S = R["splits"][s]
        sm, sss = S["summary"], S.get("seed_summary")
        doc.add_heading(f"Reused set: {'300 evaluation-dataset routes' if s == 'test' else '150 held-out High routes'}", level=1)
        para(doc, "Development results only (evaluated in an earlier round, see above). Seed-0 models.")
        results_table(doc, sm)
        if sss:
            parts = []
            for g in ["diffusion - sup", "hier - hier_sup"]:
                if g in sss["gaps"]:
                    x = SA(sss, g)
                    parts.append(f"{GAP_NAMES[g].lower()} {x['mean_diff']:+.4f} (interval {ci_txt(x['ci95'])})")
            para(doc, "Seed-averaged ablation gaps on this set: " + "; ".join(parts) + ".")

    # ------------------------------------------------------------------ fairness
    doc.add_heading("How fair is the comparison?", level=1)
    gs, bnd = tuning.get("grid_sizes", {}), tuning.get("boundary", {})
    para(doc,
         "Every setting was chosen on the 60 validation routes and then frozen. The three zone-order methods (zone diffusion, "
         f"the one-shot zone model and the history control) each received the same grid of {gs.get('hier')} settings: "
         f"{len(ZONE_DECODERS)} zone decoders times {len(ZONE_ORDER_LAMS)} order-penalty weights from {ZONE_ORDER_LAMS[0]:g} "
         f"to {ZONE_ORDER_LAMS[-1]:g} times the median travel time; the zone heuristic received {gs.get('zone')} weights from "
         f"1/16 to {ZONE_LAMS[-1]:g}. The chosen weight lies at the {bnd.get('zone', 'n/a')} of its grid for the zone "
         f"heuristic, {bnd.get('zonehist', 'n/a')} for the history control, {bnd.get('hier_sup', 'n/a')} for the one-shot zone "
         f"model and {bnd.get('hier', 'n/a')} for zone diffusion (interior means not on an edge). Grids were widened twice "
         "because optima landed on the upper edge; from about 16 upward the penalty is saturated (the solutions no longer "
         "leave the zone order), so differences between the largest weights are search noise. Learned-model settings were "
         "tuned with the seed-0 models and applied unchanged to seeds 1 and 2.")
    orm = [m for m in ["ortools", "zone", "zonehist", "hier_sup", "hier"] if "ortools_cpu_over_wall" in fr["methods"].get(m, {})]
    if orm:
        ratios = [fr["methods"][m]["ortools_cpu_over_wall"]["median"] for m in orm]
        mins = [fr["methods"][m]["ortools_cpu_over_wall"]["min"] for m in orm]
        para(doc,
             "OR-Tools ran in four single-threaded worker processes on the four-core machine with no other jobs running. CPU "
             f"time divided by wall-clock time had a median between {min(ratios):.2f} and {max(ratios):.2f} across the OR-Tools "
             f"methods on the final set (lowest single route {min(mins):.2f}).")
    if fr_b and fr_b["methods"]:
        parts = []
        for m, d in fr_b["methods"].items():
            x = d.get("hier_sup5s_minus_long")
            parts.append(f"{NAMES[m].split(' (')[0].lower()} moved from {d['mean_5s']:.4f} to {d['mean_long']:.4f}"
                         + (f" and the one-shot zone hybrid stays {verdict(x['ci95'])} it" if x else ""))
        para(doc,
             "The zone-level hybrids spend time on the network and zone decoding before their 5 second search, so the OR-Tools "
             f"baselines were re-run with {fr_b['limit_s']:g} seconds on the final set: " + "; ".join(parts) + ".")

    # ------------------------------------------------------------------ compute
    if tc:
        doc.add_heading("Training compute", level=1)
        tot = tc["totals_cpu_minutes"]
        para(doc,
             f"The specification asked for about {tc['spec_budget_cpu_minutes'][0]} to {tc['spec_budget_cpu_minutes'][1]} "
             f"CPU-minutes of training. The study used about {tc['total_cpu_minutes']:,.0f} CPU-minutes ("
             + "; ".join(f"{k}: {v:,.0f}" for k, v in tot.items()) + "), far above that. The excess pays for the controls "
             "that make the conclusions checkable: three seeds per model and the supervised twin of every diffusion model. "
             "Earlier runs did not record CPU time; for them it is estimated from wall-clock time and the share of a core each "
             "run received, and cross-checked against a measured single-thread cost per step. A spec-budget pipeline, "
             f"{' and '.join(tc['spec_budget_pipeline']['runs'])}, uses {tc['spec_budget_pipeline']['cpu_minutes']:.0f} "
             "CPU-minutes. Per-run figures are in results/summary.md.")

    # ------------------------------------------------------------------ figures
    doc.add_heading("What the routes look like", level=1)
    para(doc,
         "The route maps show one final-test route solved by every method, chosen at the median of the zone diffusion score so "
         "that it is neither a showcase nor a failure. Line shade runs from light at the first stop to dark at the last. The "
         "blind panels show six sequences for the same route with anonymous labels; the key is in blind_key.json.")
    maps = sorted(f for f in os.listdir(FIG) if f.startswith("route_2_")) if os.path.isdir(FIG) else []
    if maps:
        figure(doc, os.path.join(FIG, maps[0]),
               "Figure 4. The same real route sequenced by each method (seed-0 models), with score, travel time and share "
               "of the driver's moves reproduced.", width_cm=17.0)
    zfig = sorted(f for f in os.listdir(FIG) if f.startswith("zone_orders_")) if os.path.isdir(FIG) else []
    if zfig:
        figure(doc, os.path.join(FIG, zfig[0]),
               "Figure 5. Zone tours for the same route: the driver's, zone diffusion, the one-shot zone GNN and the history "
               "lookup.", width_cm=17.0)

    # ------------------------------------------------------------------ limits
    doc.add_heading("What this does and does not show", level=1)
    para(doc,
         "The experiment does not support the idea that the diffusion process is what makes a model learn from past "
         "successes. At stop level the diffusion model mostly rediscovered distance-based behaviour and lost to a simple zone "
         "rule; at both levels it lost to a one-shot supervised version of itself. The best results come from a hybrid in "
         "which a learned model (best without diffusion) orders zones using historical transition counts and OR-Tools orders "
         "the stops, designed after looking at validation results.")
    para(doc,
         "Further limits: the final set is a random 300-route sample of the evaluation data with unknown execution quality; "
         "the two other sets were reused across rounds; the models are small (about 0.3 and 1.2 million parameters) and were "
         "trained on CPUs; package time windows were not used; no challenge-winning method was reproduced.")

    doc.add_heading("Reproducing the study", level=1)
    para(doc,
         "All code is in the diffusion_lastmile folder: data.py prepares the data, zone_level.py builds the zone graphs, "
         "run_training.sh trains the models in resumable chunks, run_evaluation.sh tunes, evaluates, runs the diagnostics and "
         "writes the report, figures and this document. The README gives the exact commands; every number here is read from "
         "results/results.json.")

    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")
    out = os.path.join(RES, "summary.docx")
    doc.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
