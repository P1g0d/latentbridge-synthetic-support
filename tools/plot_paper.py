"""Draw Draft v3.1 figures and verify table transcriptions from persisted JSON only.

No checkpoints, hidden banks, target arrays, model imports, or network requests.
Run from any directory with: python tools/plot_paper.py
Outputs are confined to generated/figures.
"""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated/figures"
SOURCES = {}


def read_json(relative):
    raw = (ROOT / relative).read_bytes()
    SOURCES[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8"))


def number(value, digits=6):
    return f"{value:.{digits}f}".replace("-", "−")


def pm(stat, digits=6):
    return f"{number(stat['mean'], digits)} ± {number(stat['sample_sd'], digits)}"


def assert_row(text, cells):
    row = "| " + " | ".join(map(str, cells)) + " |"
    assert row.replace("−", "-") in text.replace("−", "-"), f"Table transcription differs from persisted source: {row}"


def extract_and_verify():
    id2 = read_json("results/formal/id2/results.json")["analysis"]
    xr1 = read_json("results/formal/xr1/results.json")["analysis"]
    cmr1 = read_json("results/formal/cmr1/results.json")
    cross = read_json("results/cross_support/summary.json")["campaigns"]
    functional = read_json("results/supplementary/functional_response_summary.json")["campaigns"]
    continuation = read_json("results/supplementary/continuation_summary.json")
    st = read_json("results/supplementary/feasibility_target.json")
    sta = read_json("results/supplementary/feasibility_synthetic.json")
    main = (ROOT / "paper/manuscript.md").read_text(encoding="utf-8")
    appendix = (ROOT / "paper/appendix.md").read_text(encoding="utf-8")
    # Check published gate numbers and terminal labels without constructing new bounds.
    for endpoint in ["value", "context"]:
        for value in [id2["reference_guards"][endpoint]["U95"],
                      id2["reference_guards"][endpoint]["threshold"],
                      id2["material_loss"][endpoint]["L95"],
                      id2["statistics"][endpoint]["U95"],
                      xr1["statistics"][endpoint]["qualification_bound_L95_G"],
                      xr1["statistics"][endpoint]["U95_Z"],
                      xr1["statistics"][endpoint]["L97.5_Z"],
                      cmr1["endpoints"][endpoint]["native_recovery"]["L95"],
                      cmr1["endpoints"][endpoint]["preservation"]["U95"],
                      cmr1["endpoints"][endpoint]["material_loss"]["L95"]]:
            assert number(value).replace("−", "-") in main.replace("−", "-"), f"Missing source gate value: {value}"
    assert id2["terminal"]["state"] == "MATERIAL_LOSS"
    assert cmr1["terminal_class"] == "QUALIFIED_PRESERVATION"
    formal = {}
    for campaign in ["id2", "xr1", "cmr1"]:
        formal[campaign] = {}
        for endpoint in ["value", "context"]:
            if campaign == "id2":
                s = id2["statistics"][endpoint]
                native = id2["reference_guards"][endpoint]["mean"]
                q = float(np.mean(id2["R_minus" if endpoint == "value" else "C_minus"]))
                positive = sum(v > 0 for v in id2[f"D_{endpoint}"])
                m = s["m"]
            elif campaign == "xr1":
                s = xr1["statistics"][endpoint]
                native, q = s["mean_N"], s["mean_Q"]
                positive, m = s["count_D_positive"], s["m"]
                s = {"mean": s["mean_D"], "sd": s["sd_D"]}
            else:
                e = cmr1["endpoints"][endpoint]
                native, q = e["native_mean"], e["q_pos_mean"]
                s = e["paired_difference"]
                positive, m = s["positive_count"], s["m"]
            formal[campaign][endpoint] = dict(native_mean=native, qpos_mean=q,
                                              paired_mean=s["mean"], paired_sd=s["sd"],
                                              pairs=m, positive_pairs=positive)
            if campaign == "id2":
                decision = {"statistic": "D", "mean": s["mean"],
                            "lower_97_5": id2["material_loss"][endpoint]["L95"],
                            "upper_95": id2["statistics"][endpoint]["U95"],
                            "threshold": id2["statistics"][endpoint]["delta"],
                            "source_fields": ["analysis.material_loss."+endpoint+".L95 (97.5% t critical)",
                                              "analysis.statistics."+endpoint+".U95"]}
            elif campaign == "xr1":
                xs = xr1["statistics"][endpoint]
                decision = {"statistic": "Z", "mean": xs["mean_Z"],
                            "lower_97_5": xs["L97.5_Z"], "upper_95": xs["U95_Z"], "threshold": 0.0,
                            "source_fields": ["analysis.statistics."+endpoint+".L97.5_Z",
                                              "analysis.statistics."+endpoint+".U95_Z"]}
            else:
                ce = cmr1["endpoints"][endpoint]
                decision = {"statistic": "D", "mean": ce["paired_difference"]["mean"],
                            "lower_97_5": ce["material_loss"]["L95"],
                            "upper_95": ce["preservation"]["U95"], "threshold": ce["margin"],
                            "source_fields": ["endpoints."+endpoint+".material_loss.L95 (97.5% t critical)",
                                              "endpoints."+endpoint+".preservation.U95"]}
            formal[campaign][endpoint]["decision"] = decision
            assert_row(main, [campaign.upper(), endpoint.capitalize(), m, number(native),
                              number(q), number(s["mean"]), number(s["sd"]), f"{positive}/{m}"])
        for arm in ["nat", "pos"]:
            cells = cross[campaign]["four_cell_matrix"][arm]
            assert_row(main, [campaign.upper(), "Native-trained" if arm == "nat" else "Q_pos-trained",
                              pm(cells["nat"]), pm(cells["pos"])])
        for support in ["nat", "pos"]:
            s = cross[campaign]["paired_arm_differences"][support]
            signed = f"{s['mean']:+.6f}".replace("-", "−")
            minimum = f"{s['min']:+.6f}".replace("-", "−")
            maximum = f"{s['max']:+.6f}".replace("-", "−")
            assert_row(appendix, [campaign.upper(), "Native" if support == "nat" else "Q_pos",
                                  signed, number(s["sample_sd"]), f"{minimum} to {maximum}",
                                  f"{s['positive_count']}/{s['n']}"])
        f = functional[campaign]
        for group, label in [("overall", "all positions"), ("t63", "63"), ("t127", "127")]:
            energy = f["response"]["nat"]["teacher_delta_energy"]["groups"][group]
            assert_row(appendix, [campaign.upper(), label, pm(energy["per_query_vector"], 3),
                                  pm(energy["normalized"])])
        for arm, label in [("nat", "Native"), ("pos", "Q_pos")]:
            ratios = f["response_ratios"][arm]["overall"]
            cells = [campaign.upper(), label]
            for key in ["reproduction_sse_over_teacher_delta_energy",
                        "projection_gain_onto_teacher_response", "pooled_response_cosine"]:
                s = ratios[key]
                cells.append(f"{number(s['mean'])} ({number(s['sample_sd'])})")
            assert_row(appendix, cells)
        terms = [("attention squared", "attention_error_sse"),
                 ("MLP squared" if campaign == "cmr1" else "fixed-state MLP squared",
                  "mlp_error_sse" if campaign == "cmr1" else "fixed_teacher_state_mlp_error_sse")]
        mlp_key = "mlp_error" if campaign == "cmr1" else "fixed_teacher_state_mlp_error"
        if campaign != "cmr1":
            terms.append(("propagation squared", "propagation_sse"))
        terms.append(("2〈a,m〉", f"2dot:attention_error:{mlp_key}"))
        if campaign != "cmr1":
            terms += [("2〈a,p〉", "2dot:attention_error:propagation"),
                      ("2〈m,p〉", f"2dot:{mlp_key}:propagation")]
        terms.append(("full error", "full_sse"))
        for label, key in terms:
            cells = [label]
            for cell in ["nat_on_nat", "pos_on_nat", "nat_on_pos", "pos_on_pos"]:
                s = f["decomposition"][cell][key]["groups"]["overall"]
                cells.append(f"{number(s['raw']['mean']/1e6)}; {number(s['normalized']['mean'])}")
            assert_row(appendix, cells)
        for support, label in [("nat", "Native"), ("pos", "Q_pos")]:
            cells = [campaign.upper(), label]
            for key in ["output_disagreement_energy", "2dot:e_native:d", "full_sse_difference"]:
                s = f["student_difference"][support][key]["groups"]["overall"]["normalized"]
                cells.append(f"{number(s['mean'])} ({number(s['sample_sd'])})")
            assert_row(appendix, cells)
    trajectories = {}
    for branch, b in continuation["branches"].items():
        trajectories[branch] = {"definition": b["definition"], "steps": {}}
        for step in ["0", "100", "300", "1000"]:
            s = b["steps"][step]
            audit = s["audit"]
            trajectories[branch]["steps"][step] = {
                col: audit[col]["nmse"] for col in ["nat", "pos"]}
            assert_row(appendix, [branch, step, pm(audit["nat"]["nmse"]), pm(audit["pos"]["nmse"]),
                                  number(audit["nat"]["sse"]["mean"]/1e6),
                                  number(audit["pos"]["sse"]["mean"]/1e6),
                                  number(s["full_train_objective_mse"]["mean"])])
        change = b["steps"]["1000"]["audit_change_from_step0"]
        cells = [branch]
        for col in ["nat", "pos"]:
            s = change[col]["nmse"]
            cells.append(f"{s['mean']:+.6f} ± {s['sample_sd']:.6f}".replace("-", "−"))
        assert_row(appendix, cells)
        ratios0 = b["steps"]["0"]["functional"]["ratios"]["overall"]
        ratios1 = b["steps"]["1000"]["functional"]["ratios"]["overall"]
        cells = [branch]
        for key in ["relative_response_error", "projection_gain", "cosine"]:
            cells.append(f"{number(ratios0[key]['mean'])} → {number(ratios1[key]['mean'])}")
        assert_row(appendix, cells)
    # Feasibility scores: audit and held-out results are separate stored artifacts.
    for seed in ["501", "502", "503"]:
        cells = [seed, number(st["value"][seed]["full_native"]["nmse"]),
                 number(st["context"][seed]["objects"]["full_native"]["nmse"]),
                 number(st["functional"][seed]["full_native"]["kl_mean_over_tokens"])]
        audit = sta["value"][seed]["gaussian_audit"]["gaussian"]["nmse"]
        cells.append(number(audit))
        cells += [number(st["value"][seed]["full_gaussian"]["nmse"]),
                  number(st["context"][seed]["objects"]["full_gaussian"]["nmse"]),
                  number(st["functional"][seed]["full_gaussian"]["kl_mean_over_tokens"])]
        assert_row(main + appendix, cells)
    bib = (ROOT / "paper/references.bib").read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\{([^,]+),", bib)
    # Author correspondence is not a Pandoc citation.
    citations = set(re.findall(r"(?<![\w.+-])@([a-zA-Z][a-zA-Z0-9]+)", main))
    assert len(entries) == len(set(entries)) == 18
    assert citations == set(entries), (citations - set(entries), set(entries) - citations)
    for doc in [main, appendix]:
        assert not re.search(r"\[(FIGURE|TABLE).*HERE", doc)
    return formal, cross, trajectories


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
BLUE, ORANGE = "#0072B2", "#D55E00"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def setup_figure():
    fig, ax = plt.subplots(figsize=(11.4, 8.2))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    def box(x, y, w, h, label, fill="#F2F4F7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009",
                                   facecolor=fill, edgecolor="#64748B", linewidth=1))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=14)
    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                    color="#475569", linewidth=1.3))
    ax.text(.02, .975, "Synthetic internal-operator distillation", fontsize=15, weight="bold")
    box(.04, .84, .24, .095, "iid token IDs\nL = 128")
    box(.36, .84, .26, .095, "Frozen Teacher prefix\nH = G(ids)")
    box(.72, .84, .24, .095, "Native hidden bank\ncomplete vectors")
    arrow((.29, .888), (.35, .888))
    arrow((.63, .888), (.71, .888))
    colors = ["#CCE6F4", "#FFE2BD", "#D7EACF", "#E5D9F2"]
    permutation = np.array([[2, 0, 3, 1], [0, 3, 1, 2], [3, 1, 2, 0], [1, 2, 0, 3]])
    for t in range(4):
        assert sorted(permutation[:, t]) == list(range(4))
    def bank(x, label, perm=None):
        ax.text(x+.17, .755, label, fontsize=18, weight="bold", ha="center")
        ax.text(x+.17, .716, "columns = positions; rows = examples", fontsize=15, ha="center")
        for t in range(4):
            ax.text(x+.053+t*.087, .678, f"t={t}", ha="center", fontsize=15)
        for j in range(4):
            for t in range(4):
                donor = j if perm is None else int(perm[j, t])
                left, bottom = x+.012+t*.087, .594-j*.055
                ax.add_patch(Rectangle((left, bottom), .081, .048,
                                       facecolor=colors[donor], edgecolor="#64748B", linewidth=.8))
                ax.text(left+.0405, bottom+.024, f"{chr(65+donor)}{t}", ha="center", va="center")
        ax.text(x+.17, .379, "Cell = complete hidden vector", fontsize=15, ha="center")
    bank(.04, "Native: coupled donor rows")
    bank(.59, "Q_pos: recombined rows", permutation)
    arrow((.418, .557), (.572, .557))
    ax.text(.495, .605, "Independent\ncolumn\npermutations", ha="center", va="bottom", fontsize=15)
    ax.text(.495, .468, "Same multiset\nin every column", ha="center", va="center", fontsize=15)
    ax.text(.5, .337, "Fresh Teacher labels: T(H_N) and T(H_Q)",
            ha="center", fontsize=17, weight="bold")
    box(.045, .18, .355, .115, "Native + Teacher labels\nNative-trained Student")
    box(.60, .18, .355, .115, "Q_pos + Teacher labels\nQ_pos-trained Student")
    ax.text(.5, .145, "Shared initialization • fixed learning budget", ha="center", fontsize=16)
    box(.12, .015, .76, .09, "Score both Students on common inputs\nTarget: language-induced | Dev: Native and Q_pos", "#EAF1F8")
    arrow((.14, .17), (.14, .113))
    arrow((.86, .17), (.86, .113))
    save(fig, "figure_1_setup")


def formal_figure(formal):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.5), sharey=True)
    settings = [("id2", (0.0, .078), "Material loss, both endpoints"),
                ("xr1", (-.04, .004), "Both endpoints preserved"),
                ("cmr1", (-.015, .023), "Qualified preservation")]
    for ax, (campaign, limits, outcome) in zip(axes, settings):
        xmin, xmax = limits
        for endpoint, y in [("value", 1.20), ("context", .25)]:
            d = formal[campaign][endpoint]["decision"]
            ax.plot([d["threshold"], d["threshold"]], [y-.36,y+.36], ls="--", color="#64748B", lw=1.1)
            ax.annotate("", xy=(xmax-.035*(xmax-xmin),y-.14), xytext=(d["lower_97_5"],y-.14),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
            ax.plot(d["lower_97_5"],y-.14,marker=">",color=BLUE,ms=7)
            ax.annotate("", xy=(xmin+.035*(xmax-xmin),y+.14), xytext=(d["upper_95"],y+.14),
                        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
            ax.plot(d["upper_95"],y+.14,marker="<",color=ORANGE,ms=7)
            ax.plot(d["mean"],y,"o",color="#172B4D",ms=6)
            ax.text(.03,y-.28,f"L97.5 = {d['lower_97_5']:.6f}",transform=ax.get_yaxis_transform(),color=BLUE,fontsize=15)
            ax.text(.03,y+.27,f"U95 = {d['upper_95']:.6f}",transform=ax.get_yaxis_transform(),color=ORANGE,fontsize=15)
            threshold_label = "0" if campaign=="xr1" else f"margin {d['threshold']:.3f}"
            ax.text(d["threshold"],y-.42,threshold_label,ha="center",color="#475569",fontsize=14)
        ax.set(xlim=limits,ylim=(-.28,1.65),yticks=[1.2,.25],yticklabels=["Value","Context"],
               title=campaign.upper()+" ("+str(formal[campaign]['value']['pairs'])+" pairs)",
               xlabel="Decision statistic:\n"+("Z = D - 0.05(1 - N)" if campaign=="xr1" else "D = Q - N"))
        ax.grid(axis="x",alpha=.12)
        ax.text(.5,-.45,outcome,ha="center",transform=ax.transAxes,fontsize=16,weight="bold")
    fig.suptitle("Registered one-sided confidence bounds",fontsize=20,y=.99)
    fig.text(.5,.91,"Blue: 97.5% lower bound, rightward set   |   Orange: 95% upper bound, leftward set   |   Black: mean",ha="center",fontsize=16)
    fig.subplots_adjust(left=.075,right=.99,top=.83,bottom=.32,wspace=.28)
    save(fig,"figure_2_formal_effects")


def cross_figure(cross):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.2))
    for ax, c in zip(axes, ["id2", "xr1", "cmr1"]):
        bank = cross[c]["four_cell_matrix"]
        cells = [[bank[a][b] for b in ["nat", "pos"]] for a in ["nat", "pos"]]
        values = np.array([[s["mean"] for s in row] for row in cells])
        ax.imshow(values, cmap="Blues", vmin=float(values.min())-.1*np.ptp(values),
                  vmax=float(values.max())+.2*np.ptp(values), aspect="auto")
        for row in range(2):
            for col in range(2):
                s = cells[row][col]
                # Select label contrast by the actual per-panel color normalization.
                vmin = float(values.min())-.1*np.ptp(values)
                vmax = float(values.max())+.2*np.ptp(values)
                color = "white" if (s["mean"]-vmin)/(vmax-vmin) > .60 else "#172B4D"
                ax.text(col, row, f"{s['mean']:.4f}\n± {s['sample_sd']:.4f}",
                        ha="center", va="center", fontsize=16, color=color)
        ax.set(xticks=[0, 1], xticklabels=["Native dev", "Q_pos dev"],
               yticks=[0, 1], yticklabels=["Native-trained", "Q_pos-trained"] if c=="id2" else [],
               title=f"{c.upper()} ({cross[c]['paired_replicates']} pairs)")
        d = cross[c]["paired_arm_differences"]
        ax.text(.5, -.24, f"Q − N on Native: {d['nat']['mean']:+.6f}\non Q_pos: {d['pos']['mean']:+.6f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=15)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    fig.suptitle("Common synthetic supports: mean NMSE ± sample SD\nSeparate color ranges; compare Students within each support column", fontsize=18)
    fig.subplots_adjust(left=.16, right=.99, bottom=.36, top=.78, wspace=.22)
    save(fig, "figure_3_cross_support")


def continuation_figure(trajectories):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    steps = [0, 100, 300, 1000]
    for ax, support, title in zip(axes, ["nat", "pos"], ["Native audit", "Q_pos audit"]):
        for branch in "ABCD":
            data = [trajectories[branch]["steps"][str(t)][support] for t in steps]
            color = BLUE if branch in "AB" else ORANGE
            style = "-" if branch in "AC" else "--"
            label = {"A": "A: N start / pure Q", "B": "B: N start / mixed",
                     "C": "C: Q start / pure Q", "D": "D: Q start / mixed"}[branch]
            ax.errorbar(steps, [s["mean"] for s in data], yerr=[s["sample_sd"] for s in data],
                        color=color, ls=style, marker="o", markersize=4, capsize=3, label=label)
        ax.set(title=title, xlabel="Optimizer update", ylabel="NMSE", xticks=steps)
        ax.grid(alpha=.18)
    axes[0].legend(loc="center right", fontsize=15, frameon=False)
    fig.suptitle("Exploratory continuation: all scheduled audit times (8 pairs)", fontsize=19)
    fig.tight_layout()
    save(fig, "figure_e1_continuation")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    formal, cross, continuation = extract_and_verify()
    setup_figure()
    formal_figure(formal)
    cross_figure(cross)
    continuation_figure(continuation)
    manifest = {
        "schema": "latentbridge.paper.draft_v3_1.figures.v1",
        "scope": "Static illustration and persisted-result plotting only; no model execution",
        "source_sha256": SOURCES,
        "baseline_git_commit": "bc961cbf8e41ca93fa4b173d8b4f602d5664d73b",
        "plotting": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                     "font": "DejaVu Sans", "png_dpi": 220, "vector_format": "PDF",
                     "formal_bounds": "registered one-sided L97.5 and U95, drawn separately; D for ID2/CMR1, Z for XR1",
                     "other_error_bars": "sample SD",
                     "cross_support_color_range": "separate per campaign"},
        "formal": formal,
        "cross_support": {c: {k: cross[c][k] for k in ["paired_replicates", "four_cell_matrix", "paired_arm_differences"]}
                          for c in ["id2", "xr1", "cmr1"]},
        "continuation": continuation,
        "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(OUT.glob("figure*.png")) + sorted(OUT.glob("figure*.pdf"))},
    }
    (OUT / "figure_sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("Verified persisted-source table transcriptions and 18 citation keys; drew four PNG/PDF figure pairs.")


if __name__ == "__main__":
    main()
