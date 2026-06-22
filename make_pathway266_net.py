import os
import pandas as pd

PROJECT_ROOT = "/data/home/wangzz_group/zhaipengyuan/BEPH-main"

selected_path = os.path.join(
    PROJECT_ROOT,
    "DATA_DIRECTORY/Pathway/selected_pathway_names.txt"
)

gmt_files = [
    (
        "HALLMARK",
        os.path.join(PROJECT_ROOT, "DATA_DIRECTORY/Pathway/h.all.v2026.1.Hs.symbols.gmt")
    ),
    (
        "KEGG",
        os.path.join(PROJECT_ROOT, "DATA_DIRECTORY/Pathway/c2.cp.kegg_legacy.v2026.1.Hs.symbols.gmt")
    ),
    (
        "C8",
        os.path.join(PROJECT_ROOT, "DATA_DIRECTORY/Pathway/c8.all.v2026.1.Hs.symbols.gmt")
    ),
]

out_csv = os.path.join(
    PROJECT_ROOT,
    "DATA_DIRECTORY/Pathway/pancancer_microenvironment_net_266.csv"
)

# 读取 266 个筛选通路名
with open(selected_path, "r", encoding="utf-8", errors="ignore") as f:
    selected = [x.strip() for x in f if x.strip()]

selected_set = set(selected)

rows = []
found = set()

for prefix, gmt_path in gmt_files:
    print("[INFO] Reading GMT:", gmt_path)

    if not os.path.exists(gmt_path):
        raise FileNotFoundError(f"GMT not found: {gmt_path}")

    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            term = parts[0].strip()
            genes = parts[2:]

            # 和 selected_pathway_names.txt 里的格式对齐
            source_name = f"{prefix}__{term}"

            if source_name not in selected_set:
                continue

            found.add(source_name)

            for gene in genes:
                gene = gene.strip()
                if gene:
                    rows.append({
                        "source": source_name,
                        "target": gene,
                        "weight": 1.0
                    })

net = pd.DataFrame(rows)

if net.empty:
    raise RuntimeError("生成的 net 为空，请检查 selected_pathway_names.txt 和 GMT 文件是否匹配。")

net = net.drop_duplicates(["source", "target"])
net.to_csv(out_csv, index=False)

missing = [x for x in selected if x not in found]

print("=" * 80)
print("[DONE] Saved:", out_csv)
print("[INFO] selected pathways:", len(selected))
print("[INFO] matched pathways:", len(found))
print("[INFO] missing pathways:", len(missing))
print("[INFO] net rows:", len(net))
print("=" * 80)

if missing:
    print("[WARN] Missing pathways:")
    for x in missing[:50]:
        print("  ", x)
