from scipy.stats import wilcoxon
import pandas as pd
import numpy as np


N_ATTRIBUTES = 162

model_1 = "vig"
model_2 = "mi"

results = {}
for metric_name in ["mse", "mae", "expl_var", "r2"]:
    for model_name in [model_1, model_2]:
        for class_index in range(6):
            results[f"{model_name}_{metric_name}_class_{class_index}"] = {}
        results[f"{model_name}_{metric_name}_mean"] = {}
        results[f"{model_name}_time"] = {}

for n_clusters in list(range(5, N_ATTRIBUTES, 5)):
    dfs_1, dfs_2 = [], []
    for random_state in range(20):
        df_1 = pd.read_csv(
            f"C:\\Users\\luktu\\Downloads\\gawll (1)\\results_test_frac_band_sel_estimators_4_{model_1}\\n_clusters-{n_clusters},test_size-band_sel,random_state-{random_state}.csv"
        )
        df_2 = pd.read_csv(
            f"C:\\Users\\luktu\\Downloads\\gawll (1)\\results_test_frac_band_sel_estimators_4_{model_2}\\n_clusters-{n_clusters},test_size-band_sel,random_state-{random_state}.csv"
        )
        dfs_1.append(df_1)
        dfs_2.append(df_2)

    df1 = pd.concat(dfs_1, ignore_index=True).to_dict(orient="list")
    df2 = pd.concat(dfs_2, ignore_index=True).to_dict(orient="list")

    for k, v in df1.items():
        results[f"{model_1}_{k}"][f"bands_{n_clusters}"] = v

    for k, v in df2.items():
        results[f"{model_2}_{k}"][f"bands_{n_clusters}"] = v

for metric_name in ["mse", "mae", "r2"]:
    stat_metrics = {}
    p_val_metrics = {}

    for n_bands in list(range(5, N_ATTRIBUTES, 5)):
        model_1_result = results[f"{model_1}_{metric_name}_mean"][f"bands_{n_bands}"]
        model_2_result = results[f"{model_2}_{metric_name}_mean"][f"bands_{n_bands}"]

        stat, p_val = wilcoxon(model_1_result, model_2_result)

        stat_metrics[f"{n_bands}"] = [float(stat)]
        p_val_metrics[f"{n_bands}"] = [float(p_val)]

    stat_metrics = pd.DataFrame().from_dict(stat_metrics)
    p_val_metrics = pd.DataFrame().from_dict(p_val_metrics)

    stat_metrics.to_csv(
        f"{model_1}_vs_{model_2}_wilcoxon_stat_metrics_{metric_name}_mean.csv",
        sep=";",
        float_format="%1.5f",
    )
    p_val_metrics.to_csv(
        f"{model_1}_vs_{model_2}_wilcoxon_p_val_metrics_{metric_name}_mean.csv",
        sep=";",
        float_format="%1.5f",
    )
