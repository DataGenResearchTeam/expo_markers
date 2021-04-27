import os
import re
from dataclasses import dataclass
from tqdm import tqdm
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True)


@dataclass
class TestSet:
    path: str
    inds: str

    def __post_init__(self):
        assert re.compile(r"\d+-\d+").fullmatch(self.inds) is not None
        self.str_inds = self.inds
        self.identifier = os.path.basename(os.path.dirname(self.path)) + "_" + self.inds


class EvaluationsBashWriter:
    def __init__(
        self,
        training_dir_paths,
        test_sets,
        evaluations_output_dir,
        result_bash_file_path,
        running_path,
    ):
        self.training_dir_paths = training_dir_paths
        self.test_sets = test_sets
        self.evaluations_output_dir = evaluations_output_dir
        self.result_bash_file_path = result_bash_file_path
        self.running_path = running_path

    def write(self):
        text = self._get_bash_text()
        self._write_text_to_file(text)

    def _get_bash_text(self):
        s = ""
        s += self._get_initial_string()
        for training_dir_path in tqdm(self.training_dir_paths):
            training_dir_name = os.path.basename(
                os.path.dirname(training_dir_path + "/")
            )
            s += self._get_training_dir_headlline(training_dir_path)
            weights_file_names = self._get_sorted_weights(training_dir_path)
            for weights_file_name in weights_file_names:
                for test_set in self.test_sets:
                    weights_path = os.path.join(training_dir_path, weights_file_name)
                    output_file_path = self._get_output_file_path(
                        training_dir_name, test_set.identifier, weights_file_name
                    )
                    s += self._get_single_command(
                        weights_path, test_set.path, test_set.str_inds, output_file_path
                    )
        return s

    def _get_output_file_path(
        self, training_dir_name, test_set_identifier, weights_file_name
    ):
        output_file_name = (
            "training_dir_name_"
            + training_dir_name
            + "_"
            + "test_set_identifier_"
            + test_set_identifier
            + "_"
            + "weights_"
            + weights_file_name.split(".")[0]
            + ".pickle"
        )
        output_file_path = os.path.join(self.evaluations_output_dir, output_file_name)
        return output_file_path

    def _get_initial_string(self):
        s = ""
        s += "#!/bin/bash\n"
        s += f"export PYTHONPATH={self.running_path}:$PYTHONPATH"
        s += "\n\n"
        return s

    def _get_training_dir_headlline(self, training_dir_path):
        s = ""
        s += "###############\n"
        s += "######### training dir: {}\n".format(training_dir_path)
        s += "###############\n\n"
        return s

    def _get_sorted_weights(self, training_dir_path):
        weights_file_names = sorted(
            filter(
                lambda s: re.search("model_[0-9]*.pth", s) is not None,
                os.listdir(training_dir_path),
            )
        )
        return weights_file_names

    def _get_single_command(
        self, weights_path, test_set_path, test_set_str_inds, output_file_path
    ):
        res = []
        res.append("python evaluation/eval_markers.py \\")
        res.append("MODEL.WEIGHTS {} \\".format(weights_path))
        res.append("DG_TEST_SET_PATH {} \\".format(test_set_path))
        res.append("DG_TEST_SET_INDS {} \\".format(test_set_str_inds))
        res.append("DG_EVALUATION_OUTPUT_PATH {} \\".format(output_file_path))
        s = "\n".join(res)
        s += "\n\n"
        return s

    def _write_text_to_file(self, text):
        with open(self.result_bash_file_path, "w") as f:
            f.write(text)


"""
python3 evaluation/eval_markers.py \
MODEL.WEIGHTS /expo_markers/training_logs/expo_real_1000_0_750/model_final.pth \
DG_TEST_SET_PATH /expo_markers/expo_datasets/real_image_dataset/ \
DG_TEST_SET_INDS 0-50 \
DG_EVALUATION_OUTPUT_PATH /expo_markers/evaluation_logs/test_2/t666666666666666.pickle
"""

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True)


def eval_on_given_dirs():
    training_dir_paths = [
        "/experiments/training_logs/expo_real_india_750_round000/",
        "/experiments/training_logs/expo_real_india_750_round001/",
    ]

    test_sets = [
        TestSet(path="/datasets/Expo_Real_India/", inds="750-800"),
        # TestSet(path="/datasets/Expo_Real_India/", inds="800-1000"),
        TestSet(path="/datasets/Expo_Real_DataGen_Office_250/", inds="0-50"),
        # TestSet(path="/datasets/Expo_Real_DataGen_Office_250/", inds="50-250"),
    ]

    evaluations_output_dir = "/experiments/evaluation_files/expo_paper2"

    result_bash_file_path = "./experiment1_eval_bash.sh"
    running_path = "/expo_markers"

    ebw = EvaluationsBashWriter(
        training_dir_paths,
        test_sets,
        evaluations_output_dir,
        result_bash_file_path,
        running_path,
    )
    ebw.write()


def eval_on_experiments():
    base_path = "/mnt/system1/expo_paper2/training_logs/"
    output_base_path = "/mnt/system1/expo_paper2/evaluation_files/"
    running_path = "/expo_markers"
    result_bash_files_dir = "./"
    experiments_names = os.listdir(base_path)
    experiments_names = ["map_vs_size"]
    for experiments_name in experiments_names:
        experiment_path = os.path.join(base_path, experiments_name)
        experiment_training_dir_paths = [os.path.join(experiment_path, e) for e in os.listdir(experiment_path)]
        experiment_training_dir_paths = [e for e in experiment_training_dir_paths if "8192" in e]
        test_sets = [
            TestSet(path="/datasets/Expo_Real_India/", inds="750-800"),
            TestSet(path="/datasets/Expo_Real_DataGen_Office_250/", inds="0-50"),
        ]
        evaluations_output_dir = os.path.join(output_base_path, experiments_name)
        result_bash_file_path = os.path.join(result_bash_files_dir, f"{experiments_name}__eval_bash_file.sh")
        ebw = EvaluationsBashWriter(
            experiment_training_dir_paths,
            test_sets,
            evaluations_output_dir,
            result_bash_file_path,
            running_path,
        )
        ebw.write()




if __name__ == "__main__":
    # eval_on_given_dirs()
    eval_on_experiments()




