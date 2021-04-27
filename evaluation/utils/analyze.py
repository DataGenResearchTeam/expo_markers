import pandas as pd
from collections import defaultdict
import os
import re
import pickle

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True)


class EvalFilesReader:
    def __init__(self, eval_files_path):
        self.eval_files_path = eval_files_path

    def read(self) -> pd.DataFrame:
        pickle_file_names = self._get_pickle_files()
        ddl = defaultdict(list)
        for pickle_file_name in pickle_file_names:
            train_identifier, test_identifier, iteration_num = self._parse_file_name(
                pickle_file_name
            )
            metrics = self._read_metrics_from_pickle_file(pickle_file_name)
            ddl["train_identifier"].append(train_identifier)
            ddl["test_identifier"].append(test_identifier)
            ddl["iteration_num"].append(iteration_num)
            ddl["metrics"].append(metrics)
        return pd.DataFrame(ddl)

    def _get_pickle_files(self):
        return list(
            filter(
                lambda name: name.endswith(".pickle"), os.listdir(self.eval_files_path)
            )
        )

    def _parse_file_name(self, pickle_file_name):
        pattern = r"training_dir_name_(.+)_test_set_identifier_(.+)_weights_model_(\d+).pickle"
        train_identifier, test_identifier, iteration_num = re.findall(
            pattern, pickle_file_name
        )[0]
        return train_identifier, test_identifier, int(iteration_num)

    def _read_metrics_from_pickle_file(self, pickle_file_name):
        file_path = os.path.join(self.eval_files_path, pickle_file_name)
        with open(file_path, "rb") as f:
            metrics = pickle.load(f)
        return metrics


if __name__ == "__main__":
    eval_file_reader = EvalFilesReader("/expo_markers/evaluation_files")
    eval_file_reader.read()

    "training_dir_name_expo_synt_0_750_test_set_identifier_Expo_Real_DataGen_Office_250_0-50_weights_model_0089999.pickle"
