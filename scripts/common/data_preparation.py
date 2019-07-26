import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from .utils import task2idx, pheno2idx, load_df_pickle, split_arr, write_df_pickle

def gen_equal_phenos_df(df_input):
    # Find maximum count
    phenos = df_input["phenos"]
    uniphenos, phenos_counts = np.unique(phenos, return_counts=True)
    max_counts = np.sort(phenos_counts)[3]
    for pheno, count in zip(uniphenos, phenos_counts):
        print("Pheno {}'s count = {}".format(pheno, count))
    print("Maximum count selected = {}".format(max_counts))

    # Concatenation
    df_concat_list = []
    for pheno_idx in range(13):
        df_each_pheno = df_input[df_input["phenos"] == pheno_idx].copy()
        if df_each_pheno.shape[0] == 0:
            continue
        df_each_pheno = df_each_pheno.sample(frac=1).reset_index(drop=True)
        df_each_pheno_sliced = df_each_pheno[0:max_counts]
        df_concat_list.append(df_each_pheno_sliced)

    df_concat = pd.concat(df_concat_list, axis=0, ignore_index=True).reset_index(drop=True)

    new_uniphenos, new_phenos_counts = np.unique(df_concat["phenos"], return_counts=True)
    for pheno, count in zip(new_uniphenos, new_phenos_counts):
        print("Pheno {}'s count = {}".format(pheno, count))

    print("Shape of returned df = ", df_concat.shape)
    return df_concat


def prepare_data_for_concatenated_latent(df_input_path, equal_phenos=False, output_save_path=None):
    """

    Parameters
    ----------
    df_path : str
        Pickle file for pandas.DataFrame, produced by Step 3 Feature Extraction step

    Returns
    -------

    """

    print("Preparing concatenated dataframe")
    df = load_df_pickle(df_input_path)

    # Filter data
    df_mask = (df["task_masks"] == True) & (df["pheno_masks"] == True) & (np.isnan(df["idpatients"]) == False)
    df_filtered = df[df_mask]

    if equal_phenos:
        df_filtered = gen_equal_phenos_df(df_filtered)

    # Construct dataframe
    data_dict = dict()
    unique_ids = np.unique(df_filtered["idpatients"])
    num_ids = unique_ids.shape[0]

    average_idx = 0
    grand_arr_list, grand_tasks_list, grand_idpatients_list, grand_uniphenos_list = [], [], [], []
    grand_avg_idx_list, grand_direction_list = [], []

    for patient_idx in range(num_ids):

        patient_id = unique_ids[patient_idx]
        data_dict[patient_id] = dict()
        unique_tasks = np.unique(df_filtered[df_filtered["idpatients"] == patient_id]["tasks"])
        num_uni_tasks = unique_tasks.shape[0]

        for task_id in unique_tasks:

            print("\rPatient_id = {}/{} Processing task {}/{}".format(patient_idx, num_ids, task_id, num_uni_tasks),
                  end="", flush=True)

            patient_task_mask = (df_filtered["tasks"] == task_id) & (df_filtered["idpatients"] == patient_id)
            df_each = df_filtered[patient_task_mask]
            uniphenos = np.unique(df_each["phenos"])

            split_arr_list = []
            avg_idx_list = []
            direction_list = []
            for each_id in range(df_each.shape[0]):

                # Features of each video
                fea_arr = df_each.iloc[each_id]["features"]
                split_fea_arr = split_arr(fea_arr, stride=100)
                split_arr_list.append(split_fea_arr)

                # Directions
                direction = df_each.iloc[each_id]["towards_camera"]
                direction_list.append(np.ones(split_fea_arr.shape[0]) * direction)

                # Averging identifier
                avg_idx_list.append(np.ones(split_fea_arr.shape[0])*average_idx)
                average_idx += 1

            arr_each = np.vstack(split_arr_list)
            direction_each = np.concatenate(direction_list)
            avg_idx_each = np.concatenate(avg_idx_list)
            tasks_id_arr = np.ones(arr_each.shape[0]) * task_id
            patient_id_arr = np.ones(arr_each.shape[0]) * patient_id
            uniphenos_list = [uniphenos, ] * arr_each.shape[0]

            grand_arr_list.append(arr_each)
            grand_direction_list.append(direction_each)
            grand_avg_idx_list.append(avg_idx_each)
            grand_tasks_list.append(tasks_id_arr)
            grand_idpatients_list.append(patient_id_arr)
            grand_uniphenos_list += uniphenos_list
    print()
    features = np.vstack(grand_arr_list)
    directions_all = np.concatenate(grand_direction_list)
    avg_indexes = np.concatenate(grand_avg_idx_list)
    tasks = np.concatenate(grand_tasks_list)
    idpatients = np.concatenate(grand_idpatients_list)
    df_output = pd.DataFrame(
        {"features": list(features), "tasks": tasks, "idpatients": idpatients, "phenos": grand_uniphenos_list,
         "avg_idx": list(avg_indexes), "directions":directions_all})
    if output_save_path:
        write_df_pickle(df_output, output_save_path)
        print("dataframe with shape {} written to {}".format(df_output.shape,
                                                             output_save_path))
    print("Shape of concatenated dataframe = ", df_output.shape)
    return df_output


