#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GEO 三种数据类型下载、可视化与差异表达分析

三种数据类型：
1. GSE19804 : mRNA microarray，肺癌，Tumor vs Normal
2. GSE50760 : RNA-seq，结直肠癌，Tumor/Metastasis vs Normal
3. GSE45666 : miRNA microarray，乳腺癌，Tumor vs Adjacent normal

输出：
geo_3types_results/
├── GSE19804_mRNA_microarray/
├── GSE50760_RNA_seq/
├── GSE45666_miRNA_microarray/
└── all_dataset_summary.csv
"""

import os
import re
import gzip
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import GEOparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# ===============================
# 1. 配置
# ===============================
OUT_ROOT = "./geo_3types_results"
os.makedirs(OUT_ROOT, exist_ok=True)

DATASETS = {
    "GSE19804": {
        "data_type": "mRNA_microarray",
        "description": "Lung cancer mRNA microarray: Tumor vs Normal",
        "normal_keywords": ["normal", "adjacent normal", "non-tumor", "non tumor"],
        "tumor_keywords": ["tumor", "tumour", "cancer", "carcinoma", "adenocarcinoma", "nsclc"],
    },

    "GSE50760": {
        "data_type": "RNA_seq",
        "description": "Colorectal cancer RNA-seq: Primary/metastatic cancer vs Normal colon",
        "normal_keywords": ["normal colon", "normal"],
        "tumor_keywords": [
            "primary colorectal cancer",
            "metastasized cancer",
            "metastatic",
            "metastasis",
            "cancer",
            "crc",
            "tumor",
            "tumour",
        ],
    },

    "GSE45666": {
        "data_type": "miRNA_microarray",
        "description": "Breast cancer miRNA array: Tumor vs Adjacent normal",
        "normal_keywords": ["normal", "adjacent", "adjacent normal"],
        "tumor_keywords": ["tumor", "tumour", "breast tumor", "cancer"],
    },
}

FDR_CUTOFF = 0.05
LOGFC_CUTOFF = 1.0
TOP_HEATMAP_N = 30


# ===============================
# 2. 基础函数
# ===============================
def bh_fdr(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)

    order = np.argsort(pvalues)
    ranked = pvalues[order]

    fdr = ranked * n / (np.arange(n) + 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.clip(fdr, 0, 1)

    out = np.empty(n)
    out[order] = fdr
    return out


def clean_feature_name(x):
    x = str(x)
    x = x.replace("///", "_")
    x = re.sub(r"\s+", "_", x)
    return x


def get_sample_text(gsm):
    fields = []

    for key in [
        "title",
        "source_name_ch1",
        "characteristics_ch1",
        "description",
        "treatment_protocol_ch1",
        "extract_protocol_ch1",
    ]:
        val = gsm.metadata.get(key, [])
        if isinstance(val, list):
            fields.extend([str(v) for v in val])
        else:
            fields.append(str(val))

    return " ".join(fields).lower()


def infer_label(gsm, normal_keywords, tumor_keywords):
    """
    0 = Normal
    1 = Tumor
    None = Unknown
    """
    text = get_sample_text(gsm)

    # normal 先判断，避免 adjacent normal 被 tumor 关键词误伤
    if any(k.lower() in text for k in normal_keywords):
        return 0

    if any(k.lower() in text for k in tumor_keywords):
        return 1

    return None


def save_sample_label_debug(gse, cfg, out_dir):
    labels = {}
    sample_info = []

    for gsm_id, gsm in gse.gsms.items():
        text = get_sample_text(gsm)

        label = infer_label(
            gsm,
            cfg["normal_keywords"],
            cfg["tumor_keywords"],
        )

        sample_info.append({
            "sample_id": gsm_id,
            "label": label,
            "label_name": "Normal" if label == 0 else ("Tumor" if label == 1 else "Unknown"),
            "annotation_text": text[:1000],
        })

        if label is not None:
            labels[gsm_id] = label

    df = pd.DataFrame(sample_info)
    df.to_csv(os.path.join(out_dir, "sample_label_debug.csv"), index=False)

    return labels, df


def check_gzip_file(path):
    try:
        with gzip.open(path, "rt") as f:
            _ = f.readline()
        return True
    except Exception:
        return False


def download_file(url, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    print(f"[INFO] Downloading:\n{url}")
    urllib.request.urlretrieve(url, out_path)

    return out_path


# ===============================
# 3. 芯片数据读取
# ===============================
def load_microarray_from_soft(gse_id, cfg, out_dir):
    print(f"[INFO] Downloading/loading {gse_id} by GEOparse SOFT ...")
    gse = GEOparse.get_GEO(geo=gse_id, destdir=out_dir)

    labels, _ = save_sample_label_debug(gse, cfg, out_dir)

    print("[INFO] Building expression matrix from VALUE column ...")
    expr = gse.pivot_samples("VALUE")

    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="all")
    expr = expr.fillna(expr.median(axis=1))

    common_samples = [s for s in expr.columns if s in labels]
    expr = expr[common_samples]

    y = np.array([labels[s] for s in common_samples], dtype=int)
    sample_ids = np.array(common_samples)

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"{gse_id} 没有成功识别出 Normal/Tumor 两类样本。"
            f"请查看 {out_dir}/sample_label_debug.csv"
        )

    X = expr.T.values.astype(np.float32)
    feature_ids = np.array([clean_feature_name(i) for i in expr.index.tolist()])

    if np.nanmax(X) > 100:
        print("[INFO] Applying log2(x + 1) transform ...")
        X = np.log2(X + 1.0)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[INFO] Expression matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"[INFO] Normal: {np.sum(y == 0)}, Tumor: {np.sum(y == 1)}")

    pd.DataFrame(
        X,
        index=sample_ids,
        columns=feature_ids,
    ).to_csv(os.path.join(out_dir, "expression_matrix.csv"))

    pd.DataFrame({
        "sample_id": sample_ids,
        "label": y,
        "label_name": ["Normal" if i == 0 else "Tumor" for i in y],
    }).to_csv(os.path.join(out_dir, "sample_labels.csv"), index=False)

    return X, y, sample_ids, feature_ids


# ===============================
# 4. RNA-seq count matrix 读取
# ===============================
def load_rnaseq_counts_from_ncbi(gse_id, cfg, out_dir):
    print(f"[INFO] Loading RNA-seq counts for {gse_id} ...")

    gse = GEOparse.get_GEO(geo=gse_id, destdir=out_dir)
    labels, label_debug = save_sample_label_debug(gse, cfg, out_dir)

    print("[INFO] Label recognition summary:")
    print(label_debug["label_name"].value_counts())

    count_filename = f"{gse_id}_raw_counts_GRCh38.p13_NCBI.tsv.gz"

    url_candidates = [
        (
            f"https://www.ncbi.nlm.nih.gov/geo/download/"
            f"?type=rnaseq_counts&acc={gse_id}&format=file&file={count_filename}"
        ),
        (
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:5]}nnn/"
            f"{gse_id}/suppl/{count_filename}"
        ),
    ]

    count_path = os.path.join(out_dir, count_filename)

    downloaded = False
    last_error = None

    for url in url_candidates:
        try:
            download_file(url, count_path)

            if check_gzip_file(count_path):
                downloaded = True
                break
            else:
                if os.path.exists(count_path):
                    os.remove(count_path)
                raise RuntimeError("Downloaded file is not valid gzip.")

        except Exception as e:
            last_error = e
            if os.path.exists(count_path):
                try:
                    os.remove(count_path)
                except Exception:
                    pass
            continue

    if not downloaded:
        raise RuntimeError(
            f"无法下载 {gse_id} 的 NCBI RNA-seq count matrix。最后错误: {last_error}"
        )

    print("[INFO] Reading RNA-seq raw count matrix ...")
    counts = pd.read_csv(count_path, sep="\t", compression="gzip")

    if counts.shape[1] < 3:
        raise RuntimeError(f"{count_path} 读取后列数太少，请检查文件内容。")

    gene_col = counts.columns[0]
    counts = counts.rename(columns={gene_col: "GeneID"})

    sample_cols = []
    col_to_gsm = {}

    for col in counts.columns[1:]:
        col_str = str(col)

        m = re.search(r"(GSM\d+)", col_str)
        if m:
            gsm_id = m.group(1)
        else:
            gsm_id = col_str

        if gsm_id in labels:
            sample_cols.append(col)
            col_to_gsm[col] = gsm_id

    if len(sample_cols) == 0:
        raise RuntimeError(
            f"{gse_id} count matrix 中没有找到能和 GSM 标签对应的样本列。"
        )

    expr = counts[["GeneID"] + sample_cols].copy()
    expr = expr.set_index("GeneID")

    y = np.array([labels[col_to_gsm[c]] for c in sample_cols], dtype=int)
    sample_ids = np.array([col_to_gsm[c] for c in sample_cols])
    feature_ids = np.array([clean_feature_name(i) for i in expr.index.astype(str).tolist()])

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"{gse_id} 没有成功识别出 Normal/Tumor 两类样本。"
            f"请查看 {out_dir}/sample_label_debug.csv"
        )

    X_raw = expr.T.values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    gene_sum = X_raw.sum(axis=0)
    keep = gene_sum > 10

    X_raw = X_raw[:, keep]
    feature_ids = feature_ids[keep]

    lib_size = X_raw.sum(axis=1, keepdims=True)
    cpm = X_raw / (lib_size + 1e-8) * 1e6
    X = np.log2(cpm + 1.0)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[INFO] RNA-seq matrix: {X.shape[0]} samples × {X.shape[1]} genes")
    print(f"[INFO] Normal: {np.sum(y == 0)}, Tumor: {np.sum(y == 1)}")

    pd.DataFrame(
        X,
        index=sample_ids,
        columns=feature_ids,
    ).to_csv(os.path.join(out_dir, "expression_matrix.csv"))

    pd.DataFrame({
        "sample_id": sample_ids,
        "label": y,
        "label_name": ["Normal" if i == 0 else "Tumor" for i in y],
    }).to_csv(os.path.join(out_dir, "sample_labels.csv"), index=False)

    return X, y, sample_ids, feature_ids


# ===============================
# 5. PCA
# ===============================
def plot_pca(X, y, out_dir, title):
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X_scaled)

    df = pd.DataFrame({
        "PC1": emb[:, 0],
        "PC2": emb[:, 1],
        "Group": ["Normal" if i == 0 else "Tumor" for i in y],
    })

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="Group",
        s=70,
        edgecolor="black",
        linewidth=0.3,
    )
    plt.title(title + "\nPCA")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_plot.png"), dpi=300)
    plt.close()


# ===============================
# 6. 差异表达分析
# ===============================
def differential_expression(X, y, feature_ids):
    normal = X[y == 0]
    tumor = X[y == 1]

    mean_normal = normal.mean(axis=0)
    mean_tumor = tumor.mean(axis=0)

    log2fc = mean_tumor - mean_normal

    pvals = np.ones(X.shape[1], dtype=float)

    for j in range(X.shape[1]):
        try:
            _, p = ttest_ind(
                tumor[:, j],
                normal[:, j],
                equal_var=False,
                nan_policy="omit",
            )
            if np.isnan(p):
                p = 1.0
            pvals[j] = p
        except Exception:
            pvals[j] = 1.0

    fdr = bh_fdr(pvals)

    deg_df = pd.DataFrame({
        "feature": feature_ids,
        "mean_normal": mean_normal,
        "mean_tumor": mean_tumor,
        "log2FC": log2fc,
        "pvalue": pvals,
        "fdr": fdr,
    })

    deg_df["significant"] = (
        (deg_df["fdr"] < FDR_CUTOFF) &
        (np.abs(deg_df["log2FC"]) > LOGFC_CUTOFF)
    )

    deg_df["direction"] = "NS"
    deg_df.loc[
        (deg_df["significant"]) & (deg_df["log2FC"] > 0),
        "direction"
    ] = "Up"

    deg_df.loc[
        (deg_df["significant"]) & (deg_df["log2FC"] < 0),
        "direction"
    ] = "Down"

    deg_df = deg_df.sort_values("fdr").reset_index(drop=True)

    return deg_df


# ===============================
# 7. 火山图
# ===============================
def plot_volcano(deg_df, out_dir, title):
    df = deg_df.copy()
    df["minus_log10_fdr"] = -np.log10(df["fdr"] + 1e-300)

    plt.figure(figsize=(7, 5))

    for cat, color in [("NS", "lightgrey"), ("Up", "red"), ("Down", "blue")]:
        sub = df[df["direction"] == cat]
        plt.scatter(
            sub["log2FC"],
            sub["minus_log10_fdr"],
            s=10,
            c=color,
            alpha=0.75,
            label=cat,
            linewidths=0,
        )

    plt.axvline(LOGFC_CUTOFF, linestyle="--", color="grey", linewidth=1)
    plt.axvline(-LOGFC_CUTOFF, linestyle="--", color="grey", linewidth=1)
    plt.axhline(-np.log10(FDR_CUTOFF), linestyle="--", color="grey", linewidth=1)

    plt.title(title + "\nVolcano plot")
    plt.xlabel("log2 Fold Change: Tumor - Normal")
    plt.ylabel("-log10(FDR)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "volcano_plot.png"), dpi=300)
    plt.close()


# ===============================
# 8. 热图
# ===============================
def plot_top_heatmap(X, y, sample_ids, feature_ids, deg_df, out_dir, title, top_n=30):
    sig = deg_df[deg_df["significant"]].copy()

    if sig.shape[0] == 0:
        print("[WARN] No significant features for heatmap.")
        return

    top_features = sig.sort_values("fdr").head(top_n)["feature"].tolist()

    feature_to_idx = {f: i for i, f in enumerate(feature_ids)}
    idx = [feature_to_idx[f] for f in top_features if f in feature_to_idx]

    if len(idx) == 0:
        print("[WARN] No top features matched for heatmap.")
        return

    X_sub = X[:, idx]
    X_sub = StandardScaler().fit_transform(X_sub)

    df = pd.DataFrame(
        X_sub,
        index=[f"{sid}_{'N' if lab == 0 else 'T'}" for sid, lab in zip(sample_ids, y)],
        columns=[feature_ids[i] for i in idx],
    )

    order = np.argsort(y)
    df = df.iloc[order, :]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df.T,
        cmap="vlag",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title(title + f"\nTop {len(idx)} differentially expressed features")
    plt.xlabel("Samples")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top30_heatmap.png"), dpi=300)
    plt.close()


# ===============================
# 9. 单数据集主流程
# ===============================
def process_one_dataset(gse_id, cfg):
    out_dir = os.path.join(OUT_ROOT, f"{gse_id}_{cfg['data_type']}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"[INFO] Processing {gse_id}: {cfg['description']}")
    print("=" * 80)

    if cfg["data_type"] == "RNA_seq":
        X, y, sample_ids, feature_ids = load_rnaseq_counts_from_ncbi(gse_id, cfg, out_dir)
    else:
        X, y, sample_ids, feature_ids = load_microarray_from_soft(gse_id, cfg, out_dir)

    title = f"{gse_id} ({cfg['data_type']})"

    print("[INFO] Plotting PCA ...")
    plot_pca(X, y, out_dir, title)

    print("[INFO] Differential expression analysis ...")
    deg_df = differential_expression(X, y, feature_ids)

    deg_df.to_csv(os.path.join(out_dir, "differential_expression.csv"), index=False)
    deg_df[deg_df["significant"]].to_csv(
        os.path.join(out_dir, "significant_DEGs.csv"),
        index=False,
    )

    n_up = int(np.sum(deg_df["direction"] == "Up"))
    n_down = int(np.sum(deg_df["direction"] == "Down"))

    print(f"[INFO] Significant features: Up={n_up}, Down={n_down}")

    print("[INFO] Plotting volcano plot ...")
    plot_volcano(deg_df, out_dir, title)

    print("[INFO] Plotting heatmap ...")
    plot_top_heatmap(
        X,
        y,
        sample_ids,
        feature_ids,
        deg_df,
        out_dir,
        title,
        top_n=TOP_HEATMAP_N,
    )

    print("[INFO] Top 10 DE features:")
    print(deg_df.head(10)[["feature", "log2FC", "pvalue", "fdr", "direction"]])

    return {
        "GSE": gse_id,
        "data_type": cfg["data_type"],
        "description": cfg["description"],
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_normal": int(np.sum(y == 0)),
        "n_tumor": int(np.sum(y == 1)),
        "n_up": n_up,
        "n_down": n_down,
        "out_dir": out_dir,
    }


# ===============================
# 10. 总入口
# ===============================
def main():
    all_summary = []

    for gse_id, cfg in DATASETS.items():
        try:
            summary = process_one_dataset(gse_id, cfg)
            all_summary.append(summary)
        except Exception as e:
            print(f"[ERROR] {gse_id} failed: {e}")

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(os.path.join(OUT_ROOT, "all_dataset_summary.csv"), index=False)

    print("\n" + "=" * 80)
    print("All finished.")
    print("=" * 80)
    print(summary_df)


if __name__ == "__main__":
    main()