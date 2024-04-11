from scipy.stats import wilcoxon
import pandas as pd
import numpy as np


N_ATTRIBUTES = 162

models = [
    # Analogicznie, tutaj też jeden rzad, bo affinity jest bezparameteryczne
    "affinity_prop_25_rf", "affinity_prop_25_knn",
    # VIG ours dla RF i KNNa
    "vig_ours_rf", "vig_ours_knn",
    # RF i knna na wszystkich bandach, rowniez jeden rzad
    "rf_all", "knn_all",
    # metody z ktorymi sie porównujemy
    "bombs", "mi", "random",
    ]

results = {}
for metric_name in ["mse", "mae", "expl_var", "r2"]:
    for model_name in models:
        for type_ in ["mean", "median", "min", "max", "std"]:
            results[f"{model_name}_{metric_name}_{type_}"] = []


for model_name in models:
    print(model_name)
    for metric_name in ["mse", "mae", "expl_var", "r2"]:
        print(metric_name)

        if model_name == "rf_all":
            metric_vals = []
            for random_state in range(20):
                df = pd.read_csv(
                f"C:\\Users\\luktu\\Downloads\\gawll (1)\\results_test_frac_band_sel_estimators_4_vig_ours_rf\\n_clusters-all,test_size-band_sel,random_state-{random_state}.csv"
            ,sep=",")
                metric_vals.append(df[f"{metric_name}_mean"].item())
            results[f"{model_name}_{metric_name}_mean"].append(np.mean(metric_vals))
            results[f"{model_name}_{metric_name}_std"].append(np.std(metric_vals))
            results[f"{model_name}_{metric_name}_min"].append(np.min(metric_vals))
            results[f"{model_name}_{metric_name}_max"].append(np.max(metric_vals))
            results[f"{model_name}_{metric_name}_median"].append(np.median(metric_vals))

            results[f"{model_name}_{metric_name}_mean"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_std"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_min"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_max"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_median"] += [0 for _ in list(range(5, 159, 5))]

        elif model_name == "affinity_prop_25_rf" or model_name == "affinity_prop_25_knn":
            metric_vals = []
            for random_state in range(20):
                df = pd.read_csv(
                f"C:\\Users\\luktu\\Downloads\\gawll (1)\\results_test_frac_band_sel_estimators_4_{model_name}\\n_clusters-aff,test_size-band_sel,random_state-{random_state}.csv"
            ,sep=";")    
                metric_vals.append(df[f"{metric_name}"].item())
            results[f"{model_name}_{metric_name}_mean"].append(np.mean(metric_vals))
            results[f"{model_name}_{metric_name}_std"].append(np.std(metric_vals))
            results[f"{model_name}_{metric_name}_min"].append(np.min(metric_vals))
            results[f"{model_name}_{metric_name}_max"].append(np.max(metric_vals))
            results[f"{model_name}_{metric_name}_median"].append(np.median(metric_vals))

            results[f"{model_name}_{metric_name}_mean"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_std"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_min"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_max"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_median"] += [0 for _ in list(range(5, 159, 5))]

        
        elif model_name == "knn_all":
            metric_vals = []
            for random_state in range(20):
                df = pd.read_csv(
                f"C:\\Users\\luktu\\Downloads\\gawll (1)\\results_test_frac_band_sel_estimators_4_{model_name}\\n_clusters-knn,test_size-band_sel,random_state-{random_state}.csv"
            ,sep=";")    
                metric_vals.append(df[f"{metric_name}"].item())
            results[f"{model_name}_{metric_name}_mean"].append(np.mean(metric_vals))
            results[f"{model_name}_{metric_name}_std"].append(np.std(metric_vals))
            results[f"{model_name}_{metric_name}_min"].append(np.min(metric_vals))
            results[f"{model_name}_{metric_name}_max"].append(np.max(metric_vals))
            results[f"{model_name}_{metric_name}_median"].append(np.median(metric_vals))

            results[f"{model_name}_{metric_name}_mean"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_std"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_min"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_max"] += [0 for _ in list(range(5, 159, 5))]
            results[f"{model_name}_{metric_name}_median"] += [0 for _ in list(range(5, 159, 5))]

        else:

            for n_clusters in list(range(5, N_ATTRIBUTES, 5)):
                metric_vals = []
                for random_state in range(20):
                    if model_name == "vig_ours_knn":
                        df = pd.read_csv(
                        f"C:\\Users\\luktu\\Downloads\\gawll (1)\\results_test_frac_band_sel_estimators_4_{model_name}\\n_clusters-{n_clusters},test_size-band_sel,random_state-{0}.csv"
                    , sep=";")
                    else:
                        df = pd.read_csv(
                        f"C:\\Users\\luktu\\Downloads\\gawll (1)\\results_test_frac_band_sel_estimators_4_{model_name}\\n_clusters-{n_clusters},test_size-band_sel,random_state-{random_state}.csv"
                    )
                    try:
                        metric_vals.append(df[f"{metric_name}_mean"].item())
                    except:
                        print("Trying without _mean suffix")
                        try:
                            metric_vals.append(df[f"{metric_name}"].item())
                        except:
                            raise ValueError("incorrect key")
                results[f"{model_name}_{metric_name}_mean"].append(np.mean(metric_vals))
                results[f"{model_name}_{metric_name}_std"].append(np.std(metric_vals))
                results[f"{model_name}_{metric_name}_min"].append(np.min(metric_vals))
                results[f"{model_name}_{metric_name}_max"].append(np.max(metric_vals))
                results[f"{model_name}_{metric_name}_median"].append(np.median(metric_vals))

df = pd.DataFrame.from_dict(results)
df.index = list(range(5, N_ATTRIBUTES, 5))
df.to_csv("all_reg.csv", index_label="band_index", sep=",")

