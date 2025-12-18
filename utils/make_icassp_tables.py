# -*- coding: utf-8 -*-
# save as: make_icassp_tables.py
import os, json, glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)

def df_to_markdown(df):
    # tabulate 없이 markdown 표 생성
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"]*len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pattern', default='/mnt/data/metrics_*.json')
    ap.add_argument('--out_csv', default='/mnt/data/icassp_summary_methods.csv')
    ap.add_argument('--out_perbin_csv', default='/mnt/data/icassp_perbin_all.csv')
    ap.add_argument('--out_md', default='/mnt/data/icassp_eval_summary.md')
    ap.add_argument('--plot_perbin_png', default='/mnt/data/icassp_perbin_base_vs_best.png')
    args=ap.parse_args()

    files=sorted(glob.glob(args.pattern))
    rows=[]
    for fp in files:
        m=load_metrics(fp)
        name=os.path.splitext(os.path.basename(fp))[0]
        rows.append({
            'name':name, 'N':m.get('N_test'),
            'AE':m.get('AE'), 'dlog':m.get('dlog'),
            'spread':m.get('spread_mae'), 'wet':m.get('wet_mae'),
            'gain':m.get('gain_mae'), 'room':m.get('room_mae'),
            'worst3':','.join(map(str,m.get('worst3_bins',[]))),
            'lang_en': round(m.get('lang_AE',{}).get('en', float('nan')), 2) if m.get('lang_AE') else '',
            'lang_ko': round(m.get('lang_AE',{}).get('ko', float('nan')), 2) if m.get('lang_AE') else '',
            'file':fp
        })
    if not rows:
        print("[WARN] No metrics files matched:", args.pattern)
        return

    df=pd.DataFrame(rows)
    df=df.sort_values('AE', ascending=True)
    df.to_csv(args.out_csv, index=False)

    # base/최고 성능 선택
    base_df = df[df['name'].str.contains('base', case=False)]
    best_df = df.iloc[:1]
    pick = pd.concat([base_df, best_df]).drop_duplicates('name', keep='first')

    # per-bin CSV
    perbin=[]
    for _,r in pick.iterrows():
        m=load_metrics(r['file'])
        bins=m.get('per_az12_mean', [float('nan')]*12)
        perbin.append({'name':r['name'], **{f'bin{i}':bins[i] for i in range(12)}})
    dfb=pd.DataFrame(perbin)
    dfb.to_csv(args.out_perbin_csv, index=False)

    # per-bin plot (base vs best가 모두 있을 때만)
    if len(pick)>=2 and not base_df.empty:
        m_base=load_metrics(base_df.iloc[0]['file'])
        m_best=load_metrics(best_df.iloc[0]['file'])
        b0=m_base.get('per_az12_mean', [float('nan')]*12)
        b1=m_best.get('per_az12_mean', [float('nan')]*12)
        xs=list(range(12))
        plt.figure()
        plt.plot(xs, b0, marker='o', label=base_df.iloc[0]['name'])
        plt.plot(xs, b1, marker='o', label=best_df.iloc[0]['name'])
        plt.xticks(xs, [str(i) for i in xs])
        plt.xlabel('az12 bin (0..11)')
        plt.ylabel('AE (deg)')
        plt.legend()
        plt.title('Per-bin AE: base vs best')
        plt.savefig(args.plot_perbin_png, dpi=200, bbox_inches='tight')

    # markdown 요약 (tabulate 없이 폴백)
    try:
        md = df.to_markdown(index=False)  # requires 'tabulate'
    except Exception:
        md = df_to_markdown(df)
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write("# ICASSP Eval Summary\n\n")
        f.write(md+"\n")

    print("[OK] wrote:")
    print(" -", args.out_csv)
    print(" -", args.out_perbin_csv)
    print(" -", args.out_md)
    print(" -", args.plot_perbin_png)

if __name__=='__main__':
    main()
