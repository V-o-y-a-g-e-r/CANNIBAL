from scipy.stats import wilcoxon
import pandas as pd
import numpy as np


N_ATTRIBUTES = 162

results = {}
for metric_name in ["mse", "mae", "expl_var", "r2"]:
    for class_index in range(6):
        results[f"{metric_name}_class_{class_index}"] = {}
    results[f"{metric_name}_mean"] = {}
results["time"] = {}

for n_clusters in list(range(5, N_ATTRIBUTES, 5)) + ["all"]:
    dfs = []
    for random_state in range(20):
        df = pd.read_csv(
            f"C:\\Users\\luktu\\Downloads\\gawll (1)\\results_test_frac_band_sel_estimators_4_vig\\n_clusters-{n_clusters},test_size-band_sel,random_state-{random_state}.csv"
        )
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True).to_dict(orient="list")
    for k, v in df.items():
        results[k][f"bands_{n_clusters}"] = v

for metric_name in ["mse", "mae", "r2"]:
    stat_metrics = pd.DataFrame(
        columns=list(range(5, N_ATTRIBUTES, 5)) + ["all"],
        index=list(range(5, N_ATTRIBUTES, 5)) + ["all"],
    )
    p_val_metrics = pd.DataFrame(
        columns=list(range(5, N_ATTRIBUTES, 5)) + ["all"],
        index=list(range(5, N_ATTRIBUTES, 5)) + ["all"],
    )
    for i_n_clusters in list(range(5, N_ATTRIBUTES, 5)) + ["all"]:
        for j_n_clusters in list(range(5, N_ATTRIBUTES, 5)) + ["all"]:
            if i_n_clusters == j_n_clusters:
                continue
            i = results[f"{metric_name}_mean"][f"bands_{i_n_clusters}"]
            j = results[f"{metric_name}_mean"][f"bands_{j_n_clusters}"]
            stat, p_val = wilcoxon(i, j)
            stat_metrics.loc[i_n_clusters, j_n_clusters] = float(
                np.round(stat.item(), 4)
            )
            p_val_metrics.loc[i_n_clusters, j_n_clusters] = float(
                np.round(p_val.item(), 4)
            )

    stat_metrics.to_csv(f"wilcoxon_stat_metrics_{metric_name}_mean.csv", sep=";")
    p_val_metrics.to_csv(f"wilcoxon_p_val_metrics_{metric_name}_mean.csv", sep=";")
