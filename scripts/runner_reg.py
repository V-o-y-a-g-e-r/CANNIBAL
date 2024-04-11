import os

os.environ["OMP_NUM_THREADS"] = "1"


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import SpectralClustering
from typing import List, Tuple, Dict
from time import time
from tqdm import tqdm
import pandas as pd
import numpy as np


PATH = "C:\\Users\\luktu\\Downloads\\gawll (1)\\"
TOTAL_CLASSES = 6
N_ATTRIBUTES = 162


def get_band_indices_based_on_VIGs(n_clusters: int) -> Tuple[List, Dict[int, int]]:
    vigs = []
    for i in range(TOTAL_CLASSES):
        # Those files are weighted variable interaction graphs which can be treated as adjacency matrices:
        df = pd.read_csv(
            PATH + f"class_{i}\\eVIG_dataset_urban_class_{i}_c1_a1_r1.csv", header=None
        ).values
        vigs.append(df)
    agg_vig = np.asarray(vigs).sum(axis=0)

    sc = SpectralClustering(    
        n_clusters=n_clusters, affinity="precomputed", random_state=0
    )
    labels = sc.fit_predict(agg_vig)

    cluster_representatives = {}
    for cluster_label in range(n_clusters):
        # Get indices of features in the current cluster
        cluster_indices = np.where(labels == cluster_label)[0]
        # Calculate the total similarity (sum of weights) for each feature in the cluster
        total_similarity = np.sum(agg_vig[cluster_indices], axis=1) #[[.5, .2, .342] ,.., ... ]
        # Find the index of the feature with the highest total similarity
        selected_feature_index = cluster_indices[np.argmax(total_similarity)]
        # Store the selected feature as the representative for the cluster
        assert (
            agg_vig[
                cluster_indices[np.argmax(agg_vig[cluster_indices].sum(axis=1))]
            ].sum()
            == agg_vig[selected_feature_index].sum()
        )
        cluster_representatives[cluster_label] = selected_feature_index

    cluster_representatives = {
        k: v
        for k, v in sorted(cluster_representatives.items(), key=lambda item: item[1])
    }
    bands_to_select = [cluster_representatives[i] for i in range(n_clusters)]
    return sorted(bands_to_select), cluster_representatives

def get_affinity_prop_clusters():
    from sklearn.cluster import affinity_propagation
    vigs = []
    for i in range(TOTAL_CLASSES):
        # Those files are weighted variable interaction graphs which can be treated as adjacency matrices:
        df = pd.read_csv(
            PATH + f"class_{i}\\eVIG_dataset_urban_class_{i}_c1_a1_r1.csv", header=None
        ).values
        vigs.append(df)
    agg_vig = np.asarray(vigs).sum(axis=0)
    return affinity_propagation(agg_vig)[0]

def get_bombs_bands(n_clusters) -> List:
    return (
        np.loadtxt(f"bombs/out_{n_clusters}/best_individual_bands_{n_clusters}")
        .astype(int)
        .tolist()
    )


def get_mi_bands(n_clusters) -> List:
    return np.loadtxt(f"mi/out_{n_clusters}/chosen_bands").astype(int).tolist()


# def load_data(test_size: float, random_state: int) -> Tuple[np.ndarray]:
#     data = np.load("others/urban.npy")
#     data = data.reshape(-1, 162)
#     gt = np.load("others/urban_gt.npy").T
#     X_train, X_test, y_train, y_test = train_test_split(
#         data, gt, random_state=random_state, test_size=test_size
#     )
#     return X_train, X_test, y_train, y_test

import uuid
def run(
    select_bands: bool,
    n_clusters: int,
    n_estimators: int,
    random_state: int,model
) -> None:
    start_time = time()
    X_train, X_test, y_train, y_test, test_indices = load_train_test_data()
    print("Test size: ", X_test.shape)
    print("Train size: ", X_train.shape)
    print(
        "Train frac: ", 100 * (X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]))
    )
    bands_to_select = None
    if select_bands:
        # if is_random:
        #     pass
        #     # start = np.random.randint(low=0, high=X_train.shape[1], dtype=int)
        #     # bands_to_select = list(range(start, start + n_clusters))
        # else:
        #     how = "VIG-based"
        #     print("selecting bands")
        # bands_to_select = get_bombs_bands(
        #         n_clusters=n_clusters
        #     )
        #     # bands_to_select = get_mi_bands(n_clusters)
        #     # from sklearn.decomposition import PCA
        #     # model = PCA(n_components=n_clusters, random_state=random_state)
        #     # X_train = (X_train - X_train.mean()) / X_train.std()
        #     # X_test = (X_test - X_test.mean()) / X_test.std()

        #     # model = model.fit(X_train)
        #     # X_train = model.transform(X_train)
        #     # X_test = model.transform(X_test)
        #     # print("PCA: ", X_train.shape)
        # np.random.seed(random_state)
        if model == "mi":
            bands_to_select=get_mi_bands(n_clusters=n_clusters)
        elif model == "bombs":
            bands_to_select = get_bombs_bands(n_clusters=n_clusters)
        elif model == "random":
            bands_to_select = np.random.choice(X_train.shape[-1], n_clusters, replace=False)
        
        elif model == "vig_ours_rf":
            bands_to_select, _ = get_band_indices_based_on_VIGs(n_clusters=n_clusters)
        else:
            raise ValueError()
        X_train = X_train[:, bands_to_select]
        X_test = X_test[:, bands_to_select]
        print("vig  tr: ", X_train.shape)
        print("vig  test: ", X_test.shape)
        pass
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.multioutput import MultiOutputRegressor
    if model == "random":
        print("random dis n est")
        n_estimators=2
    model = RandomForestRegressor(n_estimators==n_estimators, random_state=random_state).fit(X_train, y_train)

    # model = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators
    # ).fit(X_train, y_train)

    y_test_pred = model.predict(X_test)


    reg_metrics = {"mse": mean_squared_error(
        y_true=y_test, y_pred=y_test_pred
    ),
      "mae": mean_absolute_error(
        y_true=y_test, y_pred=y_test_pred
      ),
            "expl_var": explained_variance_score(
        y_true=y_test, y_pred=y_test_pred
    ), "r2": r2_score(
        y_true=y_test, y_pred=y_test_pred
    )}

    # class_metrics = {}
    # for k, v in metrics.items():
    #     for class_index, class_metric in enumerate(v):
    #         class_metrics[f"{k}_class_{class_index}"] = [class_metric.item()]
    # class_metrics |= {k + "_mean": [v.mean().item()] for k, v in metrics.items()}
    # del metrics
    for k, v in reg_metrics.items():
        reg_metrics[k] = [v.item()]

    reg_metrics["time"] = time() - start_time
    reg_metrics = dict(sorted(reg_metrics.items()))

    # dir_name = f"vig_knn_urban_results_test_frac_band_sel_estimators_4"
    # if not os.path.isdir(dir_name):
    #     os.makedirs(dir_name, exist_ok=True)
    # pd.DataFrame.from_dict(reg_metrics).to_csv(
    #     f"C:\\Users\\luktu\\Downloads\\gawll (1)\\{dir_name}\\n_clusters-{n_clusters},test_size-band_sel,random_state-{random_state}.csv",
    #     index=False, sep=";"
    # )
    # if select_bands:
    #     np.savetxt(
    #         f"C:\\Users\\luktu\\Downloads\\gawll (1)\\{dir_name}\\bands---n_clusters-{n_clusters},test_size-band_sel,random_state-{random_state}.txt",
    #         np.asarray(bands_to_select),
    #         delimiter=",",
    #         fmt="%d",
    #     )
    return test_indices, y_test_pred


def process_arguments(args):
    run(*args)


import uuid


def load_train_test_data():
    train_indices = []
    test_indices = None
    for class_id in range(6):
        data_all = np.load("others/urban.npy")
        data_all = data_all.reshape(-1, 162)

        gt_all = np.load("others/urban_gt.npy").T

        cond = np.where((0.2 <= gt_all[:, class_id]) & (gt_all[:, class_id] <= 0.8))[0]

        data = data_all[cond]
        gt = gt_all[cond]

        gt = gt[
            :, class_id
        ]  # Regression target is the probability of class #0, 1 - y_hat is prob. of sum of the rest classes, i.e., one vs. all

        gt = np.expand_dims(gt, 1)

        assert gt.shape[0] == data.shape[0]

        data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        data_norm = pd.DataFrame(data=data_norm)

        X_train = data_norm.sample(n=1000, replace=False, random_state=0)
        X_train_indices = X_train.index

        data_all_norm_test = (data_all - data.min(axis=0)) / (
            data.max(axis=0) - data.min(axis=0)
        )

        assert (data_all_norm_test[cond[X_train_indices]] == X_train.values).all()
        file_name = str(uuid.uuid4())
        X_train.to_csv(
            f"{file_name}.dat", sep=" ", index=False, float_format="%1.6f", header=False
        )
        X_train = np.loadtxt(f"{file_name}.dat")
        os.remove(f"{file_name}.dat")
        assert (
            X_train
            == np.loadtxt(f"data/dataset_urban_class_{class_id}.dat", skiprows=5)[
                :, :-1
            ]
        ).all()

        train_indices.extend(cond[X_train_indices].tolist())

    data_all = np.load("others/urban.npy")
    data_all = data_all.reshape(-1, 162)
    num_samples = data_all.shape[0]

    gt_all = np.load("others/urban_gt.npy").T
    train_indices = list(set(train_indices))

    all_samples_indices = np.arange(num_samples)
    test_indices = np.setdiff1d(all_samples_indices, train_indices)

    assert len(set(train_indices)) + len(set(test_indices)) == num_samples

    X_train = data_all[train_indices]
    y_train = gt_all[train_indices]

    X_test = data_all[test_indices]
    y_test = gt_all[test_indices]
    return X_train, X_test, y_train, y_test, test_indices


# def load_train_test_data():
#     train_indices = []

#     data = np.load("others/pavia.npy")
#     data = data.reshape(-1, 103)

#     gt = np.load("others/pavia_gt.npy").reshape(-1)
#     gt = np.expand_dims(gt, 1)
    
#     assert gt.shape[0] == data.shape[0]

#     data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
#     df = pd.DataFrame(data=np.hstack((data_norm, gt)), columns=list(map(str, list(range(data.shape[-1])))) + ["target"])

#     dfs = []
#     for class_idx in range(1, 10):
#         sub_df = df[df["target"] == class_idx]
#         print(f"Size for class {class_idx}: {sub_df.shape[0]}")
#         print(f"Selected size: {int(0.05 * sub_df.shape[0])}\n\n")
#         sub_df = sub_df.sample(n=int(0.05 * sub_df.shape[0]), replace=False, random_state=0)
#         dfs.append(sub_df)

#     import uuid
#     original = np.loadtxt(f"others/dataset_pavia.dat", skiprows=6)[:, :-1]
#     df = pd.concat(dfs)
#     file_name = str(uuid.uuid4())
#     df.iloc[:, :-1].to_csv(
#         f'{file_name}.dat', sep=" ", index=False,
#         float_format='%1.6f', header=False)
    
#     assert (np.sort(original.ravel()) == np.sort(np.loadtxt(f"{file_name}.dat").ravel())).all()
#     os.remove(f"{file_name}.dat")

#     train_indices = df.index.tolist()

#     data_all = np.load("others/pavia.npy")
#     data_all = data_all.reshape(-1, 103)
#     num_samples = data_all.shape[0]

#     gt_all = np.load("others/pavia_gt.npy").reshape(-1)
#     train_indices = list(set(train_indices))

#     all_samples_indices = np.arange(num_samples)
#     test_indices = np.setdiff1d(all_samples_indices, train_indices)

#     assert len(set(train_indices)) + len(set(test_indices)) == num_samples

#     X_train = data_all[train_indices]
#     y_train = gt_all[train_indices]

#     X_test = data_all[test_indices]
#     y_test = gt_all[test_indices]

#     mask = (y_test >= 1) & (y_test <= 9) & (y_test != 0)

#     y_test = y_test[mask]
#     X_test = X_test[mask]
#     return X_train, X_test, y_train, y_test, test_indices


if __name__ == "__main__":
    # pass
    # get_band_indices_based_on_VIGs(10)
    # load_train_test_data()
    # data = load_train_test_data()
    # pass
    # for n_clusters in tqdm(list(range(5, N_ATTRIBUTES, 5))):
    #     for random_state in range(5, 20):
    #         run(
    #             select_bands=True,
    #             is_random=False,
    #             n_clusters=n_clusters,
    #             n_estimators=16,
    #             test_size=0.2,
    #             random_state=random_state,
    #         )
    ##
    import multiprocessing
    arguments_list = []
    for n_clusters in tqdm(list(range(5, 162, 5))):
        arguments_list.append(
            (
                True,  # select_bands
                n_clusters,
                4,  # n_estimators
                0,
            )
        )
    num_processes = 4
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_arguments, arguments_list)

    # import multiprocessing
    # arguments_list = []
    # for random_state in range(0, 20):
    #     arguments_list.append(
            
    #             (True,  # select_bands
    #             "aff",
    #             4,  # n_estimators
    #             random_state,)
            
    #     )
    # print(arguments_list)
    # num_processes = 2
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     pool.map(process_arguments, arguments_list)
