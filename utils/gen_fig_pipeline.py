# -*- coding: utf-8 -*-
# Draw pipeline diagram with matplotlib patches (no external deps)
import argparse, os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

def add_box(ax, xy, wh, text):
    r = Rectangle(xy, wh[0], wh[1], fill=False)
    ax.add_patch(r)
    ax.text(xy[0]+wh[0]/2, xy[1]+wh[1]/2, text, ha="center", va="center")

def add_arrow(ax, p, q):
    arr = FancyArrow(p[0], p[1], q[0]-p[0], q[1]-p[1], width=0.02, length_includes_head=True)
    ax.add_patch(arr)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="figs_misc/pipeline.png")
    args=ap.parse_args()

    # ▶︎ 상위 디렉터리 자동 생성
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ax.set_xlim(0,10); ax.set_ylim(0,3); ax.axis("off")

    add_box(ax, (0.4,1.1), (1.6,0.8), "Text\n(prompt)")
    add_arrow(ax, (2.0,1.5), (2.8,1.5))
    add_box(ax, (2.8,1.1), (2.0,0.8), "Encoder\n(MiniLM/E5)")
    add_arrow(ax, (4.8,1.5), (5.6,1.5))
    add_box(ax, (5.6,1.1), (2.4,0.8), "Heads:\nReg(8D) + Az12(Arc) + Δ")
    add_arrow(ax, (8.0,1.5), (8.8,1.5))
    add_box(ax, (8.8,1.1), (1.6,0.8), "Spatial params\n[sin,cos,el,d,s,wet,g,room]")

    ax.text(5.0, 2.5, "Loss: angle + ArcMargin + Δ + align + contrast\n"
                      "Sampling: az12-weighted, dir-focus; kNN fuse at inference",
            ha="center", va="center")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

if __name__=="__main__":
    main()
