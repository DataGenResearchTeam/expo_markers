from evaluation.utils.analyze import EvalFilesReader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
from dataclasses import dataclass
import pathlib
import os
import pickle
import json
from copy import deepcopy

font_size = 20

import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True, suspend=False)

VAL_TO_TEST_MAP = {
    "Expo_Real_India_750-800": "Expo_Real_India_800-1000",
    "Expo_Real_DataGen_Office_250_0-50": "Expo_Real_DataGen_Office_250_50-250"
}

SET_TO_PATH_AND_INDS = {
    "Expo_Real_India_800-1000": ("/datasets/Expo_Real_India", "800-1000"),
    "Expo_Real_DataGen_Office_250_50-250": ("/datasets/Expo_Real_DataGen_Office_250", "50-250")
}


@dataclass
class ValTestSets:
    name: str
    val: str
    test: str


def get_df_raw(path: str, eval_type: str):


    store_path = pathlib.PurePath(path).name + "df_raw.pickle"
    if os.path.exists(store_path):
        print("TTTT Read existing")
        with open(store_path, "rb") as f:
            df = pickle.load(f)
    else:
        print("TTTT create and store")
        eval_file_reader = EvalFilesReader(path)
        df = eval_file_reader.read()
        assert eval_type in ["val", "test"]
        if eval_type == "val":
            df = df.rename(columns={"test_identifier": "val_identifier"})
        df["AP"] = [e["segm"]["AP"] for e in df.metrics]

        with open(store_path, "wb") as f:
            pickle.dump(df, f)

    return df


def get_synt_vs_real_df(df):
    def parse_train_identifier(train):
        train_set_name, round_num = re.findall(r"(\w*)_round(\d*)", train)[0]
        return train_set_name, int(round_num)

    val_test_sets = [
        ValTestSets(name="Expo_Real_DataGen_Office",
                    val="Expo_Real_DataGen_Office_250_0-50",
                    test="Expo_Real_DataGen_Office_250_50-250"),
        ValTestSets(name="Expo_Real_India",
                    val="Expo_Real_India_750-800",
                    test="Expo_Real_India_800-1000")
    ]
    ddl = defaultdict(list)
    for val_test_set in val_test_sets:
        for train_identifier in df.train_identifier.unique():
            df1 = df[df.train_identifier == train_identifier]

            df2 = df1[df1.test_identifier == val_test_set.val]
            best_iter = df2.iloc[df2["AP"].argmax()]["iteration_num"]

            val_ap = df1[(df1.test_identifier == val_test_set.val) & (df1.iteration_num == best_iter)]["AP"]
            test_ap = df1[(df1.test_identifier == val_test_set.test) & (df1.iteration_num == best_iter)]["AP"]

            train_set_name, round_num = parse_train_identifier(train_identifier)

            ddl["train"].append(train_set_name)
            ddl["val_test_set_name"].append(val_test_set.name)

            ddl["val"].append(float(val_ap))
            ddl["test"].append(float(test_ap))
            ddl["round"].append(round_num)
    df_new = pd.DataFrame(ddl)
    return df_new


def get_map_vs_size_df(df):
    def parse_train_identifier(train):
        train_set_name, size, round_num = re.findall(r"(\w*)_(\d*)_round(\d*)", train)[0]
        return train_set_name, int(size), int(round_num)

    ddl = defaultdict(list)
    for train_identifier in df.train_identifier.unique():
        df1 = df[df.train_identifier == train_identifier]
        for val_set_name in df1.test_identifier.unique():
            df2 = df1[df1.test_identifier == val_set_name]
            best_iter = df2.iloc[df2["AP"].argmax()]["iteration_num"]
            val_ap = df1[(df1.test_identifier == val_set_name) & (df1.iteration_num == best_iter)]["AP"]
            assert len(val_ap) == 1
            train_set_name, size, round_num = parse_train_identifier(train_identifier)

            ddl["train_name"].append(train_set_name)
            ddl["size"].append(size)
            ddl["val_name"].append(val_set_name)
            ddl["map"].append(float(val_ap))
            ddl["round"].append(round_num)
    df_new = pd.DataFrame(ddl)
    return df_new


def get_mix_df(df):
    def parse_train_identifier(train):
        train_set_name, round_num = re.findall(r"(.*)_round(\d*)", train)[0]
        return train_set_name, int(round_num)

    ddl = defaultdict(list)
    for train_identifier in df.train_identifier.unique():
        df1 = df[df.train_identifier == train_identifier]

        for val_set_name in df1.test_identifier.unique():
            df2 = df1[df1.test_identifier == val_set_name]
            best_iter = df2.iloc[df2["AP"].argmax()]["iteration_num"]
            val_ap = df1[(df1.test_identifier == val_set_name) & (df1.iteration_num == best_iter)]["AP"]
            train_set_name, round_num = parse_train_identifier(train_identifier)
            ddl["train_name"].append(train_set_name)
            ddl["val_name"].append(val_set_name)
            ddl["map"].append(float(val_ap))
            ddl["round"].append(round_num)
    df_new = pd.DataFrame(ddl)
    return df_new


def plot_synt_vs_real_results(df):
    plt.figure(figsize=(10, 10))
    width = 0.3
    gap = 0.02
    train_sets = df.train.unique()
    train_sets_name_map = {"expo_synt_v8_r3_750": "Synthetic", "expo_real_india_750": "Real"}
    train_sets = [train_sets_name_map[e] + f"\n({e})" for e in train_sets]
    plt.xticks(range(len(train_sets)), train_sets, size=20)
    plt.xlabel("Train Dataset", fontdict={"size": font_size})
    plt.ylabel("mAP", fontdict={"size": font_size})
    colors = ["r", "b"]
    for j, val_test_set_name in enumerate(df.val_test_set_name.unique()):

        aps_lst = []
        for i, train in enumerate(sorted(df.train.unique())):
            x_locs = [i + (j - 0.5) * width + 2 * (j - 0.5) * gap for i in range(len(df.train.unique()))]
            aps = df[(df.train == train) & (df.val_test_set_name == val_test_set_name)].test.to_list()
            aps = [e for e in aps if e != min(aps)]  ########################### Delete lowest sample from both

            aps_lst.append(aps)
            plt.scatter(len(aps) * [x_locs[i]], aps, color="g")
        avg_aps = [np.mean(e) for e in aps_lst]
        stds = [np.std(e) for e in aps_lst]
        plt.bar(x_locs, avg_aps, width=width, alpha=0.7, label=f"test_set: {val_test_set_name}")
        fontdict = {'size': 12, 'ha': 'left', 'va': 'top', 'bbox': {'fc': '0.8', 'pad': 3, "alpha": 0.4}}
        for x, y, std in zip(x_locs, avg_aps, stds):
            s = f"{y:.2f}±{std:.2f}"
            plt.text(x + 0.015, y, s, rotation=0, fontdict=fontdict)
    plt.legend(fontsize=font_size)
    plt.ylim([0, 100])


def plot_map_vs_dsize(df):
    val_names = df.val_name.unique()
    colors = ["r", "b"]
    for i, val_name in enumerate(val_names):
        df1 = df[df.val_name == val_name]
        plt.figure(figsize=(10, 7))
        plt.title(f"mAP v.s. Dataset Size\nEcalutaed on {val_name}", fontdict={"size": font_size})
        plt.xlabel("Dataset Size", fontdict={"size": font_size})
        plt.ylabel("mAP", fontdict={"size": font_size})

        train_names = df.train_name.unique()
        for j, train_name in enumerate(train_names):

            df2 = df1[df1.train_name == train_name]
            sizes = sorted(df2["size"].unique())
            aps_lst = []
            for size in sizes:
                aps = df2[
                    (df2.train_name == train_name) & (df2["size"] == size) & (df2.val_name == val_name)].map.to_list()
                aps_lst.append(aps)
                plt.scatter(len(aps) * [size], aps, color=colors[j], s=15)
            ys = [np.mean(e) for e in aps_lst]
            stds = [np.std(e) for e in aps_lst]
            plt.plot(sizes, ys, marker="*", markersize=15, c=colors[j], label=f"Trained on: {train_name}")
        #             plt.scatter(sizes, ys, s=200, c=colors[j])

        plt.legend()
        plt.grid()
        plt.show()


def plot(df):
    val_names = sorted(df.val_name.unique(), reverse=True)

    for i, val_name in enumerate(val_names):

        plt.figure(figsize=(10, 7))
        plt.title("Training with Different Mixtures of Synthetic and Real Data\n" +
                  "Expo_Synt[0:750] + Expo_Real_India[0:X] \n" +
                  f"Evaluated on {val_name}", size=font_size)

        plt.ylabel("mAP", fontdict={"size": 14})

        #### show baseline of India_750 to India_750
        if val_name == "Expo_Real_India_750-800":
            df_tmp = synt_vs_real_final_results.reset_index()
            train_india_test_india_map = float(df_tmp[(df_tmp["train"] == "expo_real_india_750") & (
                        df_tmp["val_test_set_name"] == "Expo_Real_India")].test)
            plt.plot([-100, 2000], [train_india_test_india_map, train_india_test_india_map], "--", c="r",
                     label="Training with Expo_Real_india[0:750]")
        ####

        train_sets = sorted(df.train_name.unique(), key=get_train_name_val)
        colors = ["r", "b"]
        xs = []
        aps_lst = []
        for train_name in train_sets:
            train_amount = get_train_name_val(train_name)
            xs.append(train_amount)

            aps = df[(df.train_name == train_name) & (df.val_name == val_name)].map.to_list()
            aps_lst.append(aps)
            plt.scatter(len(aps) * [train_amount], aps, marker=".", alpha=1, s=50, color="k")
        ys = [np.mean(e) for e in aps_lst]
        stds = [np.std(e) for e in aps_lst]
        val_name = "_".join(val_name.split("_")[:-1])
        plt.plot(xs, ys, marker="o", markersize=6, c="b", label="Train Set: Expo_Synt[0:750] + Expo_Real_India[0:X]")

        for x, y, std in zip(xs, ys, stds):
            s = f"{y:.2f}±{std:.2f}"
            fontdict = {'size': 12, 'ha': 'left', 'va': 'top', 'bbox': {'fc': '0.8', 'pad': 3, "alpha": 0.1}}
            plt.text(x + 7, y - 0.2, s, fontdict, rotation=-20)
        plt.ylim([78, 95])
        plt.xlim([-25, 775])

        plt.xlabel("Amount of additional real data (Expo_Real_India)", fontdict={"size": 14})
        plt.legend()
        plt.grid()

def choose_best_iteration(df):
    """
        train_identifier	            val_identifier	        iteration_num    metrics	                                        AP
    0	expo_real_india_750_round000	Expo_Real_India_750-800	18499	         {'bbox': {'AP': 95.35374142515202, 'AP50': 98....	91.699905
    1	expo_synt_v8_r3_750_round003	Expo_Real_India_750-800	25999	         {'bbox': {'AP': 81.81138519940049, 'AP50': 94....	80.754562
    """
    ddl = defaultdict(list)
    for train_identifier in df.train_identifier.unique():
        df1 = df[df.train_identifier == train_identifier]
        for val_identifier in df1.val_identifier.unique():
            df2 = df1[df1.val_identifier == val_identifier]
            best_iter_ind = df2.AP.argmax()
            best_iter = df2.iloc[best_iter_ind]["iteration_num"]
            # metrics = df2.df2.iloc[best_iter_ind]["metrics"]
            # AP = df2.df2.iloc[best_iter_ind]["AP"]

            ddl["train_identifier"].append(train_identifier)
            ddl["val_identifier"].append(val_identifier)
            ddl["iteration_num"].append(best_iter)

    return pd.DataFrame(ddl)


def get_single_eval_command(weights_path, test_set_path, test_set_inds, output_path):
    res = []
    res.append("python evaluation/eval_markers.py \\")
    res.append(f"MODEL.WEIGHTS {weights_path} \\")
    res.append(f"DG_TEST_SET_PATH {test_set_path} \\")
    res.append(f"DG_TEST_SET_INDS {test_set_inds} \\")
    res.append(f"DG_EVALUATION_OUTPUT_PATH {output_path} \\")

    s = "\n".join(res)
    s += "\n\n"
    return s


def create_map_vs_dsize_eval_bash_file(df: pd.DataFrame):
    weights_base_dir_path_map = {
        "synt": "/mnt/system1/expo_paper2/training_logs/map_vs_size/",
        "india": "/mnt/system1/expo_paper2/training_logs/map_vs_size_india/"
    }

    base_output_dir_path = "/mnt/system1/expo_paper2/test_evaluation_files/"

    txt = "# map_vs_size eval bash file\n"
    for i, row in df.iterrows():
        train_identifier = row["train_identifier"]
        val_identifier = row["val_identifier"]
        iteration_num = row["iteration_num"]
        if "synt" in train_identifier:
            weights_base_dir_path = weights_base_dir_path_map["synt"]
        elif "india" in train_identifier:
            weights_base_dir_path = weights_base_dir_path_map["india"]
        else:
            raise ValueError
        weights_path = os.path.join(weights_base_dir_path, row["train_identifier"], f"model_{iteration_num:07d}.pth")
        if not os.path.exists(weights_path):
            print(123)
        assert os.path.exists(weights_path)
        test_identifier = VAL_TO_TEST_MAP[val_identifier]
        test_set_path, test_set_inds = SET_TO_PATH_AND_INDS[test_identifier]
        output_path = os.path.join(base_output_dir_path, f"train__{train_identifier}__test__{test_identifier}__iter__{iteration_num}.pickle")
        txt += get_single_eval_command(weights_path, test_set_path, test_set_inds, output_path)
    with open("test_eval_bash_map_vs_dsize.sh", "w") as f:
        f.write(txt)


def create_mix_eval_bash_file(df: pd.DataFrame):
    weights_base_dir_path_map = {
        "mix1": "/mnt/system1/expo_paper2/training_logs/mix/",
        "mix2": "/mnt/system1/expo_paper2/training_logs/mix2/"
    }

    base_output_dir_path = "/mnt/system1/expo_paper2/test_evaluation_files/"

    txt = "# mix eval bash file\n"
    for i, row in df.iterrows():
        train_identifier = row["train_identifier"]
        val_identifier = row["val_identifier"]
        iteration_num = row["iteration_num"]
        mix_dir_name = "mix1" if "synt" in train_identifier.split("AND")[0] else "mix2"
        weights_base_dir_path = weights_base_dir_path_map[mix_dir_name]
        weights_path = os.path.join(weights_base_dir_path, row["train_identifier"], f"model_{iteration_num:07d}.pth")
        if not os.path.exists(weights_path):
            print(123)
        assert os.path.exists(weights_path)
        test_identifier = VAL_TO_TEST_MAP[val_identifier]
        test_set_path, test_set_inds = SET_TO_PATH_AND_INDS[test_identifier]
        output_path = os.path.join(base_output_dir_path, f"train__{train_identifier}__test__{test_identifier}__iter__{iteration_num}.pickle")
        txt += get_single_eval_command(weights_path, test_set_path, test_set_inds, output_path)
    with open(f"test_eval_bash_{mix_dir_name}.sh", "w") as f:
        f.write(txt)


def update_synt_vs_real_df(df):
    df["test_identifier"] = None
    df["test_metrics"] = None
    df["test_AP"] = None
    df["metrics"] = df["metrics"].apply(json.dumps)
    for train_id in df.train_identifier.unique():
        for val_id, test_id in VAL_TO_TEST_MAP.items():
            m1 = (df["train_identifier"] == train_id) & (df["val_identifier"] == val_id)
            best_loc = df[m1].AP.argmax()
            best_iter = df["iteration_num"][m1].iloc[best_loc]
            m2 = m1 & (df["iteration_num"] == best_iter)
            m3 = (df["train_identifier"] == train_id) & (df["val_identifier"] == test_id) & (df["iteration_num"] == best_iter)
            assert sum(m2) == 1 and sum(m3) == 1
            df.loc[m2, "test_identifier"] = test_id
            df.loc[m2, "test_metrics"] = df[m3]["metrics"].iloc[0]
            df.loc[m2, "test_AP"] = df[m3]["AP"].iloc[0]
    df = df.rename(columns={"metrics": "val_metrics", "AP": "val_AP"})
    # df = df[df.test_identifier.notnull()]
    df = df[df.val_identifier.isin(VAL_TO_TEST_MAP.keys())]
    ordered_columns = ["train_identifier", "iteration_num",
                       "val_identifier", "val_metrics", "val_AP",
                       "test_identifier", "test_metrics", "test_AP"]
    df = df[ordered_columns]
    return df


def update_df_with_best_iter(df, must_find_test_evaluation=True):
    dir_path = "/mnt/system1/expo_paper2/test_evaluation_files/"
    df.metrics = df.metrics.apply(json.dumps)
    df = df.rename(columns={"metrics": "val_metrics", "AP": "val_AP"})
    df["test_identifier"] = None
    df["test_metrics"] = None
    df["test_AP"] = None

    for train_id in df.train_identifier.unique():
        for val_id, test_id in VAL_TO_TEST_MAP.items():
            m1 = (df["train_identifier"] == train_id) & (df["val_identifier"] == val_id)
            best_loc = df[m1]["val_AP"].argmax()
            best_iter = df["iteration_num"][m1].iloc[best_loc]
            m2 = m1 & (df["iteration_num"] == best_iter)

            file_name = f"train__{train_id}__test__{test_id}__iter__{best_iter}.pickle"
            file_path = os.path.join(dir_path, file_name)
            if not must_find_test_evaluation and not os.path.exists(file_path):
                pass
            else:
                with open(file_path, "rb") as f:
                    d = pickle.load(f)
                df.loc[m2, "test_identifier"] = test_id
                df.loc[m2, "test_metrics"] = json.dumps(d)
                df.loc[m2, "test_AP"] = d["segm"]["AP"]
        ordered_columns = ["train_identifier", "iteration_num",
                           "val_identifier", "val_metrics", "val_AP",
                           "test_identifier", "test_metrics", "test_AP"]
        df = df[ordered_columns]
    return df


def update_mix2_with_data_from_other_experiments(map_vs_size_df, mix2_df, mix_df):
    ###########
    # update mix2_df with the following:
    #############
    # add "expo_real_india_750_xxxxx" from map_vs_size_df
    temp_df = deepcopy(map_vs_size_df[map_vs_size_df.train_identifier.apply(lambda ti: "expo_real_india_750" in ti)])

    def convert_map_vs_size__to__mix2(ti):
        # "expo_real_india_750_round001" --> "expo_real_india_0-750_AND_expo_synt_v8_r3_0-None_round001"
        round = ti.split("_round")[-1]
        res = f"expo_real_india_0-750_AND_expo_synt_v8_r3_0-None_round{round}"
        return res

    temp_df["train_identifier"] = temp_df["train_identifier"].map(convert_map_vs_size__to__mix2)
    mix2_df = pd.concat([mix2_df, temp_df])
    # add "expo_synt_v8_r3_0-750_AND_expo_real_india_0-100_round003" from mix_df
    temp_df = deepcopy(
        mix_df[mix_df.train_identifier.apply(lambda ti: "expo_synt_v8_r3_0-750_AND_expo_real_india_0-750" in ti)])

    def convert_mix_df__to__mix2(ti):
        # "expo_synt_v8_r3_0-750_AND_expo_real_india_0-750_round003" --> "expo_real_india_0-750_AND_expo_synt_v8_r3_0-750_round003"
        round = ti.split("_round")[-1]
        res = f"expo_real_india_0-750_AND_expo_synt_v8_r3_0-750_round{round}"
        return res

    temp_df["train_identifier"] = deepcopy(temp_df["train_identifier"].map(convert_mix_df__to__mix2))
    mix2_df = pd.concat([mix2_df, temp_df])
    return mix2_df


def update_synt_vs_real_with_best_mix(mix_df, synt_vs_real_df):
    ###########
    # for later: update synt_vs_real_df with data from mix (not sure yet which mixture to use)
    ###########
    temp_df = deepcopy(
        mix_df[mix_df["train_identifier"].map(lambda ti: 'expo_synt_v8_r3_0-750_AND_expo_real_india_0-750' in ti)])
    synt_vs_real_df = pd.concat([synt_vs_real_df, temp_df])
    return synt_vs_real_df


def main():
    synt_vs_real_df = get_df_raw("/mnt/system1/expo_paper2/evaluation_files/synt_vs_real/", "val")
    map_vs_size_df = get_df_raw("/mnt/system1/expo_paper2/evaluation_files/map_vs_size_combined/", "val")
    mix_df = get_df_raw("/mnt/system1/expo_paper2/evaluation_files/mix/", "val")
    mix2_df = get_df_raw("/mnt/system1/expo_paper2/evaluation_files/mix2/", "val")

    map_vs_size_best_iter_df = choose_best_iteration(map_vs_size_df)
    # mix_best_iter_df = choose_best_iteration(mix_df)
    # mix2_best_iter_df = choose_best_iteration(mix2_df)

    create_map_vs_dsize_eval_bash_file(map_vs_size_best_iter_df)
    # create_mix_eval_bash_file(mix_best_iter_df)
    # create_mix_eval_bash_file(mix2_best_iter_df)

    # synt_vs_real_df = update_synt_vs_real_df(synt_vs_real_df)
    # map_vs_size_df = update_df_with_best_iter(map_vs_size_df)
    # mix_df = update_df_with_best_iter(mix_df)
    # mix2_df = update_df_with_best_iter(mix2_df, must_find_test_evaluation=True)
    #
    # mix2_df = update_mix2_with_data_from_other_experiments(map_vs_size_df, mix2_df, mix_df)
    #
    # synt_vs_real_df = update_synt_vs_real_with_best_mix(mix_df, synt_vs_real_df)
    #
    # res = {
    #     "synt_vs_real_df": synt_vs_real_df,
    #     "map_vs_size_df": map_vs_size_df,
    #     "mix_df": mix_df,
    #     "mix2_df": mix2_df
    # }
    # return res



# synt_vs_real_df = get_synt_vs_real_df(synt_vs_real_df_raw)
# plot_synt_vs_real_results(synt_vs_real_df)
#
# map_vs_size_df = get_map_vs_size_df(map_vs_size_df_raw)
# plot_map_vs_dsize(map_vs_size_df)
#
# mix_df = get_mix_df(mix_df_raw)
# plot(mix_df)


if __name__ == "__main__":
    main()


"""
docker run -it --rm --gpus all --shm-size="8g" --env="DISPLAY" \
--net host \
-v /home/roey.ron/benchmarking/training_and_evaulation/detectron2/expo_markers:/expo_markers \
-v /mnt/system1/expo_paper2/:/mnt/system1/expo_paper2 \
-v /home/roey.ron/directed_research_playground/datasets:/datasets \
-w /expo_markers \
-e PYTHONPATH=$PYTHONPATH:/expo_markers \
--name=detectron2_debug fastai_detectron2 python evaluation/utils/analyze_results.py
"""

