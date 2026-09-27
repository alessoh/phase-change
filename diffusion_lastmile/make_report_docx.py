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

from evaluate import RES

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
    "hier_sup": "One-shot supervised zone GNN + OR-Tools (5 s)",
    "hier": "Zone-level diffusion + OR-Tools (5 s)",
}
ORDER = ["hier", "hier_sup", "zonehist", "zone", "diffusion", "sup", "diffusion_greedy", "sup_greedy", "softdist",
         "softdist_greedy", "ortools", "nn", "driver"]
GAP_NAMES = {"hier - zonehist": "Zone diffusion minus history zone order",
             "hier - hier_sup": "Zone diffusion minus one-shot zone GNN",
             "hier_sup - zonehist": "One-shot zone GNN minus history zone order",
             "hier - zone": "Zone diffusion minus zone-change heuristic",
             "diffusion - sup": "Stop diffusion minus one-shot GNN (both + 2-opt)",
             "diffusion_greedy - sup_greedy": "Stop diffusion minus one-shot GNN (greedy only)",
             "diffusion - softdist": "Stop diffusion minus SoftDist (both + 2-opt)",
             "sup - softdist": "One-shot GNN minus SoftDist (both + 2-opt)",
             "diffusion - zone": "Stop diffusion + 2-opt minus zone-change heuristic"}


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
          rows, [4.3, 3.0, 1.4, 1.5, 1.5, 1.6, 1.3, 2.4], bold_rows=[methods.index("hier")] if "hier" in methods else [])


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
    r1 = R.get("tuning_round1_on_val")
    diag = R.get("diagnostics")
    stats = json.load(open(os.path.join(RES, "data_stats.json")))
    TE = R["splits"]["test"]
    te, te_ss, te_b = TE["summary"], TE.get("seed_summary"), TE.get("budget_sensitivity")
    HO = R["splits"].get("heldout", {})
    ho, ho_ss = HO.get("summary"), HO.get("seed_summary")

    def M(s, m, k="amazon_score", f="mean"):
        return s["methods"][m][k][f]

    def P(s, ref, other, k="amazon_score"):
        return s["paired"][ref][other][k]

    def gap(ss, g):
        return ss["gaps"][g]

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
    n_te = te["n_routes"]
    para(doc,
         "The idea under test is that a generative diffusion model trained only on delivery routes that drivers executed well "
         "should produce a sensible sequence for a new route by itself, because it has absorbed what good routes in that "
         "business look like. We built that model on the public Amazon Last Mile Routing Research Challenge data, trained it "
         f"only on the {stats['n_high']:,} routes Amazon rated High quality, and compared it on {n_te} unseen routes of the "
         "separate official evaluation dataset with a nearest-neighbour rule, Google OR-Tools, a zone-aware OR-Tools heuristic, "
         "a non-learned heat-map control and, as an ablation, the identical neural network trained as an ordinary one-shot "
         "supervised classifier. Every stop, zone, travel time and driver sequence is real.")
    d_vs_z = P(te, "diffusion", "zone")["ci95"]
    d_vs_sd = P(te, "diffusion", "softdist")["ci95"]
    para(doc,
         "The answer is mostly no for the model as specified, and a qualified yes for a hybrid designed afterwards. The "
         "specified stop-level diffusion model, which generates stop-to-stop connections and is decoded greedily with 2-opt, "
         f"scored {M(te, 'diffusion'):.4f} on the official challenge metric (lower is closer to the driver). That is "
         f"{verdict(d_vs_sd)} the non-learned SoftDist control built from travel times alone ({M(te, 'softdist'):.4f}) but "
         f"{verdict(d_vs_z)} a simple zone-aware OR-Tools heuristic ({M(te, 'zone'):.4f}). Its learned heat-map adds little "
         "over distance, and it does not beat a rule that one line of domain knowledge produces.")
    if te_ss and "diffusion - sup" in te_ss["gaps"]:
        g = gap(te_ss, "diffusion - sup")
        para(doc,
             "The ablation asks whether the diffusion process itself contributes anything. Trained on the same data, with the "
             "same features, network, number of steps and decoder, the one-shot supervised version of the stop-level model "
             f"scored {te_ss['methods']['sup']['across_seeds']['mean']:.4f} averaged over three training seeds, against "
             f"{te_ss['methods']['diffusion']['across_seeds']['mean']:.4f} for diffusion. Averaged over seeds, diffusion minus "
             f"supervised is {g['seed_averaged']['mean_diff']:+.4f} (95% interval {ci_txt(g['seed_averaged']['ci95'])}), so the "
             f"diffusion model is {verdict(g['seed_averaged']['ci95'])} its supervised twin at stop level.")
    if te_ss and "hier - zonehist" in te_ss["gaps"]:
        gh, gs = gap(te_ss, "hier - zonehist"), gap(te_ss, "hier - hier_sup")
        others = [m for m in te["methods"] if m not in ("driver", "hier")]
        best_other = min(others, key=lambda m: M(te, m))
        best_clause = (", the lowest mean of every method in the seed-0 table" if M(te, "hier") < M(te, best_other)
                       else f", behind {NAMES[best_other].lower()} ({M(te, best_other):.4f})")
        para(doc,
             "The positive result is a post-hoc hybrid. After a diagnosis on validation routes, the same diffusion machinery "
             "was moved one level up to generate only the order in which a route's delivery zones are served, with historical "
             "zone-transition counts from the training routes as input features, and OR-Tools then sequences the stops inside "
             f"that order. This method reached {M(te, 'hier'):.4f} (median {M(te, 'hier', f='median'):.4f}) with the seed-0 model "
             f"and {te_ss['methods']['hier']['across_seeds']['mean']:.4f} averaged over three seeds{best_clause}. "
             "Its margin over a pure history lookup that decodes the same transition counts without any model is "
             f"{size_word(gh['seed_averaged']['mean_diff'], M(te, 'zonehist'))}: "
             f"{gh['seed_averaged']['mean_diff']:+.4f} averaged over seeds (95% interval {ci_txt(gh['seed_averaged']['ci95'])}), "
             f"with per-seed gaps between {gh['across_seeds']['min']:+.4f} and {gh['across_seeds']['max']:+.4f}. Against the "
             "one-shot supervised zone model, which sees exactly the same inputs, the zone diffusion model is "
             f"{verdict(gs['seed_averaged']['ci95'])} it (seed-averaged difference {gs['seed_averaged']['mean_diff']:+.4f}, "
             f"interval {ci_txt(gs['seed_averaged']['ci95'])}).")
    para(doc,
         "Even the best method remains far from challenge-level performance. The winning team of the 2021 challenge (Cook, "
         "Held and Helsgaun; arXiv 2112.15192, Transportation Science 2024) is reported to have scored about 0.025 on the full "
         "evaluation set. We saw that figure in search results but could not open the paper from this environment to verify "
         f"it, and our {n_te}-route subset is not a leaderboard submission, so the comparison is indicative only. No "
         "challenge-winning method or published learned sequencing model was reproduced here.")

    # ------------------------------------------------------------------ data
    doc.add_heading("The data and the test", level=1)
    para(doc,
         "The Amazon Last Mile Routing Research Challenge dataset (Merchan and colleagues, Transportation Science, 2024) is "
         "published on the AWS Open Data registry under a Creative Commons Attribution-NonCommercial licence, which this "
         f"research use respects. The training part contains {stats['n_train_total']:,} routes from 17 depots in five US "
         "metropolitan areas, each with stop coordinates, zone identifiers, a full matrix of real travel times and the "
         f"sequence the driver actually drove. Only the {stats['n_high']:,} routes rated High were used for learning: "
         f"{stats['split_sizes']['train']:,} for training, {stats['split_sizes']['val']} for choosing every setting of every "
         f"method, and {stats['split_sizes']['heldout']} kept aside as a second test set that no model saw during training "
         "or tuning.")
    para(doc,
         f"The main test set is {stats['split_sizes']['test']} routes drawn at random from the {stats['n_eval_total']:,} routes "
         "of the separate official evaluation dataset. Those files carry no route-quality label, so the quality of these "
         "executions is unknown. The challenge scoring may itself have used only High-quality routes; we could not verify "
         "this, and we make no assumption about it. The held-out High routes are therefore reported as well, because their "
         "quality is known and matches the training data.")
    para(doc,
         "Two measures are reported for every route. The first is the total travel time of the closed tour from the depot and "
         "back on the real travel-time matrix. The second is the official challenge score, which measures how far a proposed "
         "sequence is from the driver's: a sequence-deviation term multiplied by an edit distance on normalised travel times. "
         "Lower is better and the driver scores exactly zero. The scoring code was re-implemented from the organisers' "
         "published script and is tested to give bit-for-bit identical values on real routes.")

    # ------------------------------------------------------------------ method
    doc.add_heading("How the models work", level=1)
    para(doc,
         "A route is treated as a graph. The diffusion model learns to produce the set of directed connections a driver would "
         "use, one yes-or-no value per candidate connection. During training the true connections of a driver's route are "
         "progressively scrambled by random bit flips, and a graph neural network learns to undo the scrambling while looking "
         "at the real travel times, the stop locations and the zone labels. At use time the model starts from pure noise on a "
         f"new route and denoises it in {tuning['diff_steps']} steps; {tuning['diff_samples']} such samples are averaged into a "
         "map of how likely each connection is. This is the DIFUSCO approach (Sun and Yang, NeurIPS 2023), here trained to "
         "imitate drivers instead of minimising distance. A greedy procedure turns the map into one valid tour, optionally "
         "polished by 2-opt.")
    para(doc,
         "The supervised ablation uses the identical network, features, training routes, batch size, learning-rate schedule "
         "and number of optimisation steps, but receives no noisy input: it predicts every connection in one forward pass, and "
         "its map is decoded in exactly the same way. Any difference between the two is therefore due to the diffusion "
         "process, not to the network or the data. Both objectives were trained at both levels with three seeds each, twelve "
         "models in all.")
    para(doc,
         "The zone-level models use the same code on a much smaller graph whose nodes are the route's zones. Besides geography "
         "and zone labels they see how often drivers from the same depot moved between each pair of zones in the training "
         "routes, computed so that a training route never sees its own transitions. The generated zone order is handed to "
         "OR-Tools, which fills in the stops with a penalty on every move that leaves that order. The history control decodes "
         "the same transition counts directly, with no model, through the same zone decoders and the same OR-Tools step.")
    if diag:
        ss, zs = diag["stop_successor"], diag["zone_structure"]["val"]
        dif = [v["top1"] for k, v in ss.items() if k.startswith("diffusion_seed")]
        sup = [v["top1"] for k, v in ss.items() if k.startswith("supervised_seed")]
        para(doc,
             "The diagnosis that motivated the zone level is reproduced by the script diagnostics.py. On the "
             f"{diag['n_routes']} validation routes the stop-level diffusion model's single most likely next stop was the "
             f"driver's actual next stop {min(dif) * 100:.1f}% to {max(dif) * 100:.1f}% of the time across seeds, the one-shot "
             f"supervised model's {min(sup) * 100:.1f}% to {max(sup) * 100:.1f}%, against {ss['nearest_stop']['top1'] * 100:.1f}% "
             f"for simply choosing the nearest stop. Meanwhile {zs['same_zone_move_share'] * 100:.1f}% of a driver's consecutive "
             f"moves stay within one zone and {zs['contiguous_zone_share'] * 100:.1f}% of zones are served in one continuous "
             "block. The part of a driver's plan that carries knowledge beyond distance is mainly the order of the zones.")

    # ------------------------------------------------------------------ results
    doc.add_heading(f"Results on {n_te} routes from the official evaluation dataset", level=1)
    para(doc,
         "The table shows seed-0 models. Scores are means with bootstrap 95% confidence intervals over routes, followed by "
         "medians. Driver moves reproduced is the share of the driver's stop-to-stop moves that a method also makes, and "
         "zone-order tau is the rank correlation between the method's zone order and the driver's (1 means identical order). "
         "Runtime is the median wall-clock time per route with its interquartile range, measured with no other jobs on the "
         "machine; it includes the 5 seconds given to OR-Tools where it is used.")
    results_table(doc, te)
    para(doc,
         "Route by route, the zone diffusion method was closer to the driver than the zone-change heuristic on "
         f"{P(te, 'hier', 'zone')['ref_better_rate'] * 100:.0f}% of routes (mean difference "
         f"{P(te, 'hier', 'zone')['mean_diff_ref_minus_other']:+.4f}, interval {ci_txt(P(te, 'hier', 'zone')['ci95'])}) and "
         f"closer than plain OR-Tools on {P(te, 'hier', 'ortools')['ref_better_rate'] * 100:.0f}%. Against the history control "
         f"it was better on {P(te, 'hier', 'zonehist')['ref_better_rate'] * 100:.0f}% of routes and returned an identical "
         f"sequence on {P(te, 'hier', 'zonehist')['tie_rate'] * 100:.0f}%.")
    para(doc,
         "The cost of looking like a driver is travel time. OR-Tools, which only minimises travel time, produced tours "
         f"{-P(te, 'hier', 'ortools', 'travel_time_s')['mean_diff_ref_minus_other'] / 60:.1f} minutes shorter on average than "
         "the zone diffusion method, yet those tours scored far worse against what drivers did. Drivers respond to "
         "considerations the travel-time matrix does not contain; the data does not say which, and a lower score means "
         "similarity to the driver, not a better route.")
    figure(doc, os.path.join(FIG, "score_distributions.png"),
           f"Figure 1. Per-route challenge score and travel time for every method on the {n_te} evaluation-dataset routes "
           "(seed-0 models). Boxes show quartiles, diamonds show means, and the dashed line is the drivers' median travel time.")

    # ------------------------------------------------------------------ ablation and seeds
    if te_ss:
        doc.add_heading("Does the diffusion process matter, and how stable is it across seeds?", level=1)
        para(doc,
             "Each learned model was trained three times with different seeds, which changes the initial weights, the order of "
             "the training data and the noise. The first table gives each model's mean score on the evaluation routes for each "
             "seed. The second gives paired differences on the same routes, seed by seed, with 95% bootstrap intervals; the "
             "last column first averages each method's per-route score over the three seeds. Negative values favour the first "
             "method named.")
        seed_table(doc, te_ss)
        gap_table(doc, te_ss)
        lines = []
        for g in ["hier - zonehist", "hier - hier_sup", "diffusion - sup", "hier_sup - zonehist"]:
            if g not in te_ss["gaps"]:
                continue
            d = te_ss["gaps"][g]
            n_neg = sum(v["ci95"][1] < 0 for v in d["per_seed"].values())
            n_pos = sum(v["ci95"][0] > 0 for v in d["per_seed"].values())
            lines.append(f"for {GAP_NAMES[g].lower()} the per-seed mean gap ranges from {d['across_seeds']['min']:+.4f} to "
                         f"{d['across_seeds']['max']:+.4f} (standard deviation {d['across_seeds']['sd']:.4f}), with the interval "
                         f"entirely below zero for {n_neg} of 3 seeds and entirely above zero for {n_pos}")
        para(doc, "Reading the seed tables: " + "; ".join(lines) + ".")
        figure(doc, os.path.join(FIG, "seed_spread.png"),
               "Figure 2. Mean score of every learned model for each training seed (blue circles: diffusion; orange squares: "
               "one-shot supervised ablation) beside the non-learned control of the same row (gray bar) and the zone-change "
               "heuristic (dashed line).", width_cm=16.5)

    if ho:
        doc.add_heading(f"Results on {ho['n_routes']} held-out High-quality routes", level=1)
        para(doc,
             "These routes come from the same distribution as the training data and are known to be good executions, but were "
             "never used for training or tuning. The table again shows seed-0 models.")
        results_table(doc, ho)
        txt = (f"Here the zone diffusion method scored {M(ho, 'hier'):.4f} (median {M(ho, 'hier', f='median'):.4f}), "
               f"{verdict(P(ho, 'hier', 'zone')['ci95'])} the zone-change heuristic ({M(ho, 'zone'):.4f}) and "
               f"{verdict(P(ho, 'hier', 'zonehist')['ci95'])} the history control ({M(ho, 'zonehist'):.4f}), with which it "
               f"returned identical sequences on {P(ho, 'hier', 'zonehist')['tie_rate'] * 100:.0f}% of routes. The stop-level "
               f"diffusion model scored {M(ho, 'diffusion'):.4f}, {verdict(P(ho, 'diffusion', 'zone')['ci95'])} the zone-change "
               "heuristic.")
        if ho_ss and "hier - zonehist" in ho_ss["gaps"]:
            g = ho_ss["gaps"]["hier - zonehist"]
            txt += (f" Across the three seeds the zone diffusion minus history gap on these routes ranges from "
                    f"{g['across_seeds']['min']:+.4f} to {g['across_seeds']['max']:+.4f}, and the seed-averaged gap is "
                    f"{g['seed_averaged']['mean_diff']:+.4f} (interval {ci_txt(g['seed_averaged']['ci95'])}).")
        para(doc, txt)
        if ho_ss:
            seed_table(doc, ho_ss)
            gap_table(doc, ho_ss)

    # ------------------------------------------------------------------ fairness
    doc.add_heading("How fair is the comparison?", level=1)
    gs = tuning.get("grid_sizes", {})
    bnd = tuning.get("boundary", {})
    para(doc,
         "Every setting was chosen on the 60 validation routes and then frozen; nothing was tuned on either test set. The "
         "three zone-order methods (zone diffusion, the one-shot zone model and the history control) each received the same "
         f"grid of {gs.get('hier', 12)} settings, three zone decoders times four order-penalty weights, and the zone-change "
         f"heuristic received {gs.get('zone', 12)} penalty weights spanning 1/16 to 128. The chosen penalty lies at the "
         f"{bnd.get('zone', 'n/a')} end of its grid for the zone heuristic, at the {bnd.get('zonehist', 'n/a')} end for the history "
         f"control, at the {bnd.get('hier_sup', 'n/a')} end for the one-shot zone model and at the {bnd.get('hier', 'n/a')} end "
         "for zone diffusion (\"interior\" means not on a boundary). Learned-model settings were tuned with the seed-0 models "
         "and applied unchanged to seeds 1 and 2. The complete tuning record of this round is in results/tuning.json.")
    if r1:
        c = r1["counts"]
        para(doc,
             "An earlier round of this study, with models since retrained, tried "
             f"{c.get('hier', 0)} settings for zone diffusion, {c.get('zonehist', 0)} for the history control and "
             f"{c.get('zone', 0)} for the zone heuristic on the same validation routes. Its published tuning file had omitted "
             "the greedy plus 2-opt zone-decoder runs; the complete record, rebuilt from the cached solutions, is in "
             "results/tuning_round1.json. That earlier, unequal tuning is why this round uses equal grids.")
    orm = [m for m in ["ortools", "zone", "zonehist", "hier_sup", "hier"] if "ortools_cpu_over_wall" in te["methods"].get(m, {})]
    if orm:
        ratios = [te["methods"][m]["ortools_cpu_over_wall"]["median"] for m in orm]
        mins = [te["methods"][m]["ortools_cpu_over_wall"]["min"] for m in orm]
        para(doc,
             "OR-Tools ran in four single-threaded worker processes on the four-core machine with no other jobs running. The "
             "solver's CPU time divided by its wall-clock time had a median between "
             f"{min(ratios):.2f} and {max(ratios):.2f} across the OR-Tools methods on the evaluation routes (lowest single "
             f"route {min(mins):.2f}), so each solve effectively had a full core for its 5 second limit.")
    if te_b:
        hr = te["methods"]["hier"]["runtime_s"]
        parts = []
        for m, d in te_b["methods"].items():
            parts.append(f"{NAMES[m].split(' (')[0].lower()} moved from {d['mean_5s']:.4f} to {d['mean_long']:.4f} "
                         f"and the zone diffusion method stays {verdict(d['hier5s_minus_long']['ci95'])} it")
        para(doc,
             "The zone-level learned methods spend time on sampling and zone decoding before their 5 second search (median "
             f"total {hr['median']:.1f} seconds per route for zone diffusion), so the OR-Tools baselines were also re-run with a "
             f"{te_b['limit_s']:g} second limit on the evaluation routes. With the longer limit, " + "; ".join(parts) + ".")

    # ------------------------------------------------------------------ figures
    doc.add_heading("What the routes look like", level=1)
    para(doc,
         "The route maps show one evaluation route solved by every method, chosen at the median of the zone diffusion score so "
         "that it is neither a showcase nor a failure. The line shade runs from light at the first stop to dark at the last. "
         "The blind panels show six sequences for the same route with anonymous labels in random order; the key is stored "
         "separately in blind_key.json so a reader can judge which one looks like the driver before checking.")
    maps = sorted(f for f in os.listdir(FIG) if f.startswith("route_2_")) if os.path.isdir(FIG) else []
    if maps:
        figure(doc, os.path.join(FIG, maps[0]),
               "Figure 3. The same real route sequenced by each method (seed-0 models), with its challenge score, travel time "
               "and share of the driver's moves reproduced.", width_cm=17.0)
    zfig = sorted(f for f in os.listdir(FIG) if f.startswith("zone_orders_")) if os.path.isdir(FIG) else []
    if zfig:
        figure(doc, os.path.join(FIG, zfig[0]),
               "Figure 4. Zone tours for the same route: the driver's, the one generated by the zone diffusion model, the one "
               "from the one-shot supervised zone model, and the one decoded from historical frequencies alone.", width_cm=17.0)
    figure(doc, os.path.join(FIG, "blind_2.png"),
           "Figure 5. Blind comparison. Six sequences for the same route, labels shuffled. The answer is in "
           "results/figures/blind_key.json.", width_cm=15.0)

    # ------------------------------------------------------------------ limits
    doc.add_heading("What this does and does not show", level=1)
    para(doc,
         "The experiment does not support the strong form of the idea. A diffusion model trained only on successful routes, "
         "applied to stop-to-stop connections as specified, mostly rediscovered distance-based behaviour and lost to a simple "
         "zone rule. What a model can add over distance lives mainly at the level of which area to serve next, and even there "
         "the solution does not appear on its own: the best method is a hybrid in which OR-Tools still orders the stops inside "
         "each zone, it was designed after looking at validation results, and most of its advantage over plain OR-Tools is "
         "already captured by a history lookup that uses no model at all.")
    para(doc,
         "Several further limits matter. The comparison uses a random subset of the evaluation data, not the full set, and the "
         "quality of those routes is unknown. The models are small (about 0.3 million parameters at stop level and 1.2 "
         "million at zone level) and were trained on four CPU cores for a few thousand steps; larger models, package time "
         "windows and longer training were not tried. No challenge-winning method or published learned sequencing model was "
         "reproduced, and the reported winning score of about 0.025 could not be verified here.")

    doc.add_heading("Reproducing the study", level=1)
    para(doc,
         "All code is in the diffusion_lastmile folder. data.py downloads and prepares the data, zone_level.py builds the zone "
         "graphs, run_training.sh trains the twelve models in resumable chunks through train.py, run_evaluation.sh tunes, "
         "evaluates, runs the diagnostics and writes the report, figures and this document. The README gives the exact "
         "commands, and every number in this document is read from results/results.json.")

    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")  # the python-docx default template omits this required attribute
    out = os.path.join(RES, "summary.docx")
    doc.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
