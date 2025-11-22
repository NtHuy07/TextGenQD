import hydra
from omegaconf import OmegaConf

from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset

import json
import os
import pickle
from pathlib import Path

from openelm import ELM
from openelm.evaluation_metrics import get_metric_names


@hydra.main(
    config_name="elmconfig",
    version_base="1.2",
)
def main(config):
    config.qd.output_dir = config.output_dir
    config.env.output_dir = config.output_dir
    config.model.output_dir = config.output_dir
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)

    rng = np.random.default_rng(config.env.seed)
    dataset = load_dataset(config.env.dataset, config.env.dataset_ver, trust_remote_code=True)

    test_set = dataset['test']
    data_index = rng.integers(low=0, high=len(test_set) - 1, size=config.env.num_test_set)
    test_set = np.array(test_set)[data_index]
    print(data_index)
    

    config.env.behavior_space = get_behavior_space(config=config)

    elm = ELM(config)


    log_names = ["Data id", "Overall/Max fitness", "Overall/Min fitness", "Overall/Mean fitness", "Overall/QD Score", "Overall/Coverage"]
    if config.qd.qd_name == "ga":
        log_names.extend(["Overall/Max GA fitness", "Overall/Mean GA fitness", "Overall/Min GA fitness"])
    
    for name in get_metric_names():
        log_names.extend([f"Overall/QD {name}", f"Overall/Max {name}", f"Overall/Mean {name}", f"Overall/Min {name}",
                          f"Overall/Avg QD {name}", f"Overall/Avg Max {name}", f"Overall/Avg Mean {name}", f"Overall/Avg Min {name}",
                          f"Overall/Std QD {name}", f"Overall/Std Max {name}", f"Overall/Std Mean {name}", f"Overall/Std Min {name}"
                    ])

    
    log_df = pd.DataFrame(columns=log_names)

    history = []
    fitness_history = dict()
    fitness_history["max"] = []
    fitness_history["min"] = []
    fitness_history["mean"] = []
    fitness_history["qd_score"] = []
    fitness_history["coverage"] = []
    if config.qd.qd_name == "ga":
        fitness_history["max_ga"] = []
        fitness_history["mean_ga"] = []
        fitness_history["min_ga"] = []


    for name in get_metric_names():
        fitness_history[f"max_{name}"] = []
        fitness_history[f"mean_{name}"] = []
        fitness_history[f"min_{name}"] = []
        fitness_history[f"qd_{name}"] = []

    for id, test_data in enumerate(tqdm(test_set, desc="Data: ")):
        
        best_indv, log_info, history_ = elm.run(init_steps=config.qd.init_steps, total_steps=config.qd.total_steps, data=test_data, data_id=id)

        print("Performance:", log_info)
        print("Best Individual: ", best_indv)
        
        total_info, fitness_history = log(log_info, fitness_history, config, id)

        history.append(history_)
        log_df = save_results(config=config, log_info=history, log_df=log_df, total_info=total_info, step=id)
    



def log(log_info, fitness_history, config, id):

    fitness_history["max"].append(log_info[f"Data {id}/Max fitness"])
    fitness_history["min"].append(log_info[f"Data {id}/Min fitness"])
    fitness_history["mean"].append(log_info[f"Data {id}/Mean fitness"])
    fitness_history["qd_score"].append(log_info[f"Data {id}/QD Score"])
    fitness_history["coverage"].append(log_info[f"Data {id}/Coverage"])

    total_info = {"Data_id": id,
                "Overall/Max fitness": log_info[f"Data {id}/Max fitness"], 
                "Overall/Min fitness": log_info[f"Data {id}/Min fitness"],
                "Overall/Mean fitness": log_info[f"Data {id}/Mean fitness"],
                "Overall/QD Score": log_info[f"Data {id}/QD Score"],
                "Overall/Coverage": log_info[f"Data {id}/Coverage"],
    }
    
    if config.qd.qd_name == "ga":
        fitness_history["max_ga"].append(log_info[f"Data {id}/Max GA fitness"])
        fitness_history["min_ga"].append(log_info[f"Data {id}/Min GA fitness"])
        fitness_history["mean_ga"].append(log_info[f"Data {id}/Mean GA fitness"])
        total_info["Overall/Max GA fitness"] = log_info[f"Data {id}/Max GA fitness"]
        total_info["Overall/Mean GA fitness"] = log_info[f"Data {id}/Mean GA fitness"]
        total_info["Overall/Min GA fitness"] = log_info[f"Data {id}/Min GA fitness"]

    for name in get_metric_names():
        fitness_history[f"max_{name}"].append(log_info[f"Data {id}/Max {name}"])
        fitness_history[f"max_{name}"].append(log_info[f"Data {id}/Mean {name}"])
        fitness_history[f"min_{name}"].append(log_info[f"Data {id}/Min {name}"])
        fitness_history[f"qd_{name}"].append(log_info[f"Data {id}/QD {name}"])
        total_info[f"Overall/QD {name}"] = log_info[f"Data {id}/QD {name}"]
        total_info[f"Overall/Max {name}"] = log_info[f"Data {id}/Max {name}"]
        total_info[f"Overall/Mean {name}"] = log_info[f"Data {id}/Mean {name}"]
        total_info[f"Overall/Min {name}"] = log_info[f"Data {id}/Min {name}"]

    return total_info, fitness_history

def get_behavior_space(config):
    return np.array(config.env.behavior_space)[np.array(config.env.behavior)]


def save_results(config, log_info, log_df, total_info, step: int):
    # create folder for dumping results and metadata
    output_folder = Path(config.output_dir) / f"step_{step}"
    global_output_folder = Path(config.output_dir)
    os.makedirs(output_folder, exist_ok=True)

    with open((output_folder / "fitness_history.pkl"), "wb") as f:
        pickle.dump(log_info, f)

    log_df = pd.concat([log_df, pd.DataFrame(total_info, index=[0])], axis=0, ignore_index=True)
    log_df.to_csv((global_output_folder / "log.csv"))

    # save env_name to check later, for verifying correctness of environment to run with snapshot load
    tmp_config = dict()
    tmp_config["env_name"] = config.env.env_name

    with open((output_folder / "config.json"), "w") as f:
        json.dump(tmp_config, f)
    f.close()

    return log_df


if __name__ == "__main__":
    main()
    