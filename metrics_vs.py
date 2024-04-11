from scipy.stats import wilcoxon
import pandas as pd
import numpy as np


N_ATTRIBUTES = 103

model_1 = "vig_ours_rf"
# random
# bombs
out_dict = {}

for model_2 in ["bombs", "mi", "random"]:

    results = {}
    for metric_name in ["f1_weighted", "precision_weighted", "recall_weighted"]:
        for model_name in [model_1, model_2]:
            results[f"{model_name}_{metric_name}"] = {}

    for n_clusters in list(range(5, N_ATTRIBUTES, 5)):
        dfs_1, dfs_2 = [], []
        for random_state in range(20):
            df_1 = pd.read_csv(
                f"C:\\Users\\luktu\\Downloads\\gawll (1)\\{model_1}_pavia_results_test_frac_band_sel_estimators_4\\n_clusters-{n_clusters},test_size-band_sel,random_state-{random_state}.csv"
            , sep=";")[["f1_weighted", "precision_weighted", "recall_weighted"]]
            df_2 = pd.read_csv(
                f"C:\\Users\\luktu\\Downloads\\gawll (1)\\{model_2}_pavia_results_test_frac_band_sel_estimators_4\\n_clusters-{n_clusters},test_size-band_sel,random_state-{random_state}.csv"
            , sep=";")[["f1_weighted", "precision_weighted", "recall_weighted"]]
            dfs_1.append(df_1)
            dfs_2.append(df_2)

        df1 = pd.concat(dfs_1, ignore_index=True).to_dict(orient="list")
        df2 = pd.concat(dfs_2, ignore_index=True).to_dict(orient="list")

        for k, v in df1.items():
            results[f"{model_1}_{k}"][f"bands_{n_clusters}"] = v

        for k, v in df2.items():
            results[f"{model_2}_{k}"][f"bands_{n_clusters}"] = v


    for metric_name in ["f1_weighted", "precision_weighted", "recall_weighted"]:
        p_val_metrics = []

        for n_bands in list(range(5, N_ATTRIBUTES, 5)):
            model_1_result = results[f"{model_1}_{metric_name}"][f"bands_{n_bands}"]
            model_2_result = results[f"{model_2}_{metric_name}"][f"bands_{n_bands}"]

            stat, p_val = wilcoxon(model_1_result, model_2_result)

            p_val_metrics.append(float(p_val))


        out_dict[f"{metric_name}_{model_2}"] = p_val_metrics


p_val_metrics = pd.DataFrame().from_dict(out_dict)
index = list(range(5, N_ATTRIBUTES, 5))
p_val_metrics.index = index

p_val_metrics.to_csv(
    f"pavia_vs.csv",
    sep=";",
    float_format="%1.3f", index_label="Band"
)
