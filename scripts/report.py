"""Render learning curves, retention matrix and research report."""
import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path("results")
NAMES = dict(frozen="Frozen", finetune="Fine-tuning", replay_128="Replay 128", replay_512="Replay 512",
             ewc="EWC", distill="Distillation", hybrid_512="Hybrid 512", gated_512="Gated replay 512",
             joint="Joint reference")


def select(data, order, name):
    return [r for r in data["runs"] if r["order"] == order and r["model"] == name]


def figures(data):
    fig, panels = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    names = ("finetune", "replay_128", "replay_512", "gated_512", "joint")
    for order, ax in enumerate(panels):
        for name, color in zip(names, ("#b16f46", "#6486a8", "#088b75", "#8764a5", "#314a5f")):
            rows = select(data, order, name)
            samples = np.array([[s["macro_f1"] for s in row["stages"]] for row in rows])
            ax.plot(range(1,samples.shape[1]+1), samples.mean(0), "o-", color=color,
                    label=NAMES[name], markersize=4)
        ax.set(xlabel="Family-learning episode", title=f"Order {order+1}", ylim=(0,1))
        ax.grid(alpha=.25)
        ax.spines[["top","right"]].set_visible(False)
    panels[0].set_ylabel("Macro-F1 across announced classes")
    fig.legend(*panels[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0,0,1,.88))
    fig.savefig(ROOT/"retention-curves.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)

    fig, panels = plt.subplots(1, 2, figsize=(10.5, 3.5), sharey=True)
    for ax, mode in zip(panels, ("finetune","replay_512")):
        runs = select(data, 0, mode)
        labels = runs[0]["classes"][:7]
        a = np.array([[mean(s["stages"][i]["by_class"].get(name, np.nan) for s in runs)
                       for i in range(6)] for name in labels])
        im = ax.imshow(a, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set(title=NAMES[mode],xlabel="Episode",xticks=range(6),xticklabels=range(1,7),
               yticks=range(7),yticklabels=labels)
    fig.colorbar(im, ax=panels, shrink=.85, label="Held-out recall")
    fig.savefig(ROOT/"family-retention.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def report(data):
    dark = colors.HexColor("#19374c")
    fonts = Path(matplotlib.get_data_path())/"fonts"/"ttf"
    pdfmetrics.registerFont(TTFont("StudySans",str(fonts/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("StudySans-Bold",str(fonts/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("StudySans",normal="StudySans",bold="StudySans-Bold")
    styles = getSampleStyleSheet()
    for k in ("Normal","BodyText"):
        styles[k].fontName = "StudySans"
    for k in ("Title","Heading2"):
        styles[k].fontName = "StudySans-Bold"
        styles[k].textColor = dark
    styles["Title"].alignment=0
    styles["Title"].fontSize=22
    styles["Title"].leading=27
    styles["BodyText"].leading=14
    styles["BodyText"].spaceAfter=8
    body=[]
    values={(r["order"],r["model"]):r["metrics"] for r in data["summary"]}

    def p(txt, style="BodyText"):
        body.append(Paragraph(txt,styles[style]))

    def section(title,txt):
        p(title,"Heading2")
        p(txt)

    def stat(name, metric, order=0):
        return values[order,name][metric]["mean"]

    def plot(name,height):
        body.append(Image(str(ROOT/name),width=487,height=height,kind="proportional"))

    p("Continual Learning for Emerging Threat Families","Title")
    p("A replay, regularization and open-set recognition study on UNSW-NB15")
    p(f"<b>Abstract.</b> Across {len(data['runs'])} runs, two family sequences and five seeds, "
      "we measure how model updates preserve earlier knowledge. On the primary sequence, "
      f"512-row replay reduces forgetting from {stat('finetune','forgetting')*100:.2f} to "
      f"{stat('replay_512','forgetting')*100:.2f} percentage points and raises final macro-F1 "
      f"from {stat('finetune','final_macro_f1'):.3f} to {stat('replay_512','final_macro_f1'):.3f}. "
      "The open-set results are weak on three permanently unseen attack categories.")
    section("1. Research questions",
            "Does limited memory retain old families without slowing new family learning? "
            "Can regularization or knowledge distillation close the gap to cumulative training? "
            "Can class-distance or model confidence support permanent unknown-family rejection?")
    section("2. Related work",
            "Elastic weight consolidation estimates parameter importance [1]. Learning without "
            "forgetting uses previous outputs to guide future logits [2]. Small replay buffers "
            "provide a practical alternative [3]. Open-set recognition addresses detection beyond "
            "the training label inventory [4]. We evaluate each mechanism within the same stream.")
    section("3. Data and study design",
            "Use the published UNSW-NB15 feature partitions [5] with 16 flow-local variables. "
            "Remove exact duplicate features and train-test overlaps. Fit preprocessing only "
            "on benign and Fuzzers warmup rows. Five families arrive later, following two orders. "
            "Backdoor, Shellcode and Worms remain in test only. The source partitions provide "
            "no timestamp evidence for the reconstructed class order.")
    section("4. Model and policy",
            "The small MLP shares a fixed-capacity head. Expose only introduced classes. "
            "Class-balanced replay allocates equal shares by label. Online EWC uses squared "
            "per-example gradients and a discounted diagonal estimate. Distillation constrains "
            "relative old-class logits. An open-set gate reduces training from seven to two "
            "epochs on incoming batches with fewer than 24% novel observations.")
    body.append(PageBreak())

    section("5. Retention and transfer",
            "Forgetting is peak minus final recall on nonterminal classes. Backward transfer "
            "is final minus introduction recall. The few-shot forward probe fits a separate "
            "binary head and measures AP change from random to learned representations.")
    rows=[["Policy","Final F1","Forgetting","Backward transfer","Update time (ms)"]]
    for name, label in NAMES.items():
        rows.append([label,f"{stat(name,'final_macro_f1'):.3f}",f"{stat(name,'forgetting'):.3f}",
                     f"{stat(name,'backward_transfer'):+.3f}",
                     f"{stat(name,'mean_update_seconds')*1000:.1f}"])
    t=Table(rows,colWidths=[130,69,83,101,104],hAlign="LEFT")
    t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"StudySans"),
                           ("FONTNAME",(0,0),(-1,0),"StudySans-Bold"),
                           ("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(1,0),(-1,-1),"RIGHT"),
                           ("BACKGROUND",(0,0),(-1,0),dark),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                           ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#edf4f7")]),
                           ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    body.append(t)
    plot("retention-curves.png",183)
    p("<b>Figure 1.</b> Mean held-out macro-F1 in both predefined class orders.")
    p("The 512-row replay arm outperforms the hybrid arm on final macro-F1 in both orders. "
      "The gate reduces the primary-order update budget from 42 to 17 epochs and reduces "
      "final macro-F1. Training times measure only updates on one local CPU environment.")
    body.append(PageBreak())

    section("6. Case analysis", "Per-family trajectories show that a frozen model can report zero "
            "forgetting while failing to learn new labels. A retention metric needs its accompanying "
            "adaptation measure. Bounded replay spreads limited capacity across more families as they arrive.")
    plot("family-retention.png",192)
    p("<b>Figure 2.</b> Mean test recall by class and episode. Empty cells precede class introduction.")
    section("7. Unknown-family recognition",
            f"With 512-row replay in the primary sequence, maximal softmax uncertainty yields "
            f"AUROC {stat('replay_512','confidence_auc'):.3f}. The diagonal centroid baseline "
            f"yields AUROC {stat('replay_512','centroid_auc'):.3f} and rejects "
            f"{stat('replay_512','centroid_recall')*100:.2f}% of unknown-family observations "
            "at the known calibration threshold. The three held-out families overlap with "
            "known families in these coarse features. A nominal rejection budget on known "
            "examples alone does not ensure useful detection of new threats.")
    section("8. Limits and conclusion",
            "This is a constructed class-incremental comparison on a single cyber-range corpus. "
            "Held-out family names proxy previously unseen categories. Different protocols, "
            "memory selection, or a more expressive representation could change the results. "
            "Prioritize independent time-ordered datasets and cleaner calibration before "
            "considering practical deployments.")
    p("References","Heading2")
    p("[1] Kirkpatrick et al., PNAS (2017). [2] Li and Hoiem, ECCV (2016). "
      "[3] Chaudhry et al. (2019). [4] Bendale and Boult, CVPR (2016). "
      "[5] Moustafa and Slay, MilCIS (2015). Full citations appear in docs/METHODOLOGY.md.")

    def footer(c,doc):
        c.setFont("StudySans",8)
        c.setFillColor(dark)
        c.drawString(52,27,"Continual Learning for Emerging Threat Families")
        c.drawRightString(554,27,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/"continual-threat-detection-report.pdf"),
                          title="Continual Learning for Emerging Threat Families",
                          leftMargin=52,rightMargin=52,topMargin=42,bottomMargin=42)
    doc.build(body,onFirstPage=footer,onLaterPages=footer)


if __name__=="__main__":
    with gzip.open(ROOT/"study.json.gz","rt",encoding="utf8") as source:
        data=json.load(source)
    figures(data)
    report(data)
