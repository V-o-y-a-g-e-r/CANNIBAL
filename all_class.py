from scipy.stats import wilcoxon
import pandas as pd
import numpy as np


N_ATTRIBUTES = 103

models = [
    # metody z ktorymi sie porównujemy
    "bombs", "mi", "random",
    # VIG ours dla RF i KNNa
    "vig_ours_rf", "vig_ours_knn",
    # GAWLL - tutaj bierzemy 40 bandow ktore on wykrył i tylko jes jeden rząd w tabeli
    "rf_gawll_sel_40", "knn_gawll_sel_40",
    # RF i knna na wszystkich bandach, rowniez jeden rzad
    "rf_all", "knn_all",
    # Analogicznie, tutaj też jeden rzad, bo affinity jest bezparameteryczne
    "affinity_prop_22_rf", "affinity_prop_22_knn"
    ]
mmm = "acc;avg_acc;f1_macro;f1_micro;f1_weighted;precision_macro;precision_micro;precision_weighted;recall_macro;recall_micro;recall_weighted;time"
mmm = mmm.split(";")


results = {}
for metric_name in mmm:
    for model_name in models:
        for type_ in ["mean", "median", "min", "max", "std"]:
            results[f"{model_name}_{metric_name}_{type_}"] = []


for model_name in models:
    print(model_name)
    for metric_name in mmm:
        print(metric_name)

        if model_name == "rf_gawll_sel_40" or model_name == "knn_gawll_sel_40" or model_name == "affinity_prop_22_rf" or model_name == "affinity_prop_22_knn":
            metric_vals = []
            for random_state in range(20):
                if "aff" in model_name:
                    sub = "aff"
                else:
                    sub = "knn"
                df = pd.read_csv(
                f"C:\\Users\\luktu\\Downloads\\gawll (1)\\{model_name}_pavia_results_test_frac_band_sel_estimators_4\\n_clusters-{sub},test_size-band_sel,random_state-{random_state}.csv"
            ,sep=";")    
                metric_vals.append(df[f"{metric_name}"].item())
            results[f"{model_name}_{metric_name}_mean"].append(np.mean(metric_vals))
            results[f"{model_name}_{metric_name}_std"].append(np.std(metric_vals))
            results[f"{model_name}_{metric_name}_min"].append(np.min(metric_vals))
            results[f"{model_name}_{metric_name}_max"].append(np.max(metric_vals))
            results[f"{model_name}_{metric_name}_median"].append(np.median(metric_vals))

            results[f"{model_name}_{metric_name}_mean"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_std"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_min"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_max"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_median"] += [0 for _ in list(range(5, 99, 5))]
            

        elif model_name == "rf_all":
            metric_vals = []
            for random_state in range(20):
                df = pd.read_csv(
                f"C:\\Users\\luktu\\Downloads\\gawll (1)\\vig_ours_rf_pavia_results_test_frac_band_sel_estimators_4\\n_clusters-all,test_size-band_sel,random_state-{random_state}.csv"
            ,sep=";")    
                metric_vals.append(df[f"{metric_name}"].item())
            results[f"{model_name}_{metric_name}_mean"].append(np.mean(metric_vals))
            results[f"{model_name}_{metric_name}_std"].append(np.std(metric_vals))
            results[f"{model_name}_{metric_name}_min"].append(np.min(metric_vals))
            results[f"{model_name}_{metric_name}_max"].append(np.max(metric_vals))
            results[f"{model_name}_{metric_name}_median"].append(np.median(metric_vals))

            results[f"{model_name}_{metric_name}_mean"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_std"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_min"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_max"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_median"] += [0 for _ in list(range(5, 99, 5))]


        elif model_name == "knn_all":
            metric_vals = []
            for random_state in range(20):
                df = pd.read_csv(
                f"C:\\Users\\luktu\\Downloads\\gawll (1)\\knn_all_pavia_results_test_frac_band_sel_estimators_4\\n_clusters-aff,test_size-band_sel,random_state-{random_state}.csv"
            ,sep=";")    
                metric_vals.append(df[f"{metric_name}"].item())
            results[f"{model_name}_{metric_name}_mean"].append(np.mean(metric_vals))
            results[f"{model_name}_{metric_name}_std"].append(np.std(metric_vals))
            results[f"{model_name}_{metric_name}_min"].append(np.min(metric_vals))
            results[f"{model_name}_{metric_name}_max"].append(np.max(metric_vals))
            results[f"{model_name}_{metric_name}_median"].append(np.median(metric_vals))

            results[f"{model_name}_{metric_name}_mean"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_std"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_min"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_max"] += [0 for _ in list(range(5, 99, 5))]
            results[f"{model_name}_{metric_name}_median"] += [0 for _ in list(range(5, 99, 5))]
        else:
            for n_clusters in list(range(5, N_ATTRIBUTES, 5)):
                metric_vals = []
                for random_state in range(20):
                    df = pd.read_csv(
                    f"C:\\Users\\luktu\\Downloads\\gawll (1)\\{model_name}_pavia_results_test_frac_band_sel_estimators_4\\n_clusters-{n_clusters},test_size-band_sel,random_state-{random_state}.csv"
                ,sep=";")    
                    metric_vals.append(df[f"{metric_name}"].item())
                results[f"{model_name}_{metric_name}_mean"].append(np.mean(metric_vals))
                results[f"{model_name}_{metric_name}_std"].append(np.std(metric_vals))
                results[f"{model_name}_{metric_name}_min"].append(np.min(metric_vals))
                results[f"{model_name}_{metric_name}_max"].append(np.max(metric_vals))
                results[f"{model_name}_{metric_name}_median"].append(np.median(metric_vals))

                

            

df = pd.DataFrame.from_dict(results)
df.index = list(range(5, N_ATTRIBUTES, 5))
df.to_csv("all_clf.csv", index_label="band_index", sep=",")

