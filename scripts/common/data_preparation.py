import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from .utils import task2idx, pheno2idx, load_df_pickle, split_arr


def prepare_data_for_concatenated_latent(df_path):
    """

    Parameters
    ----------
    df_path : str
        Pickle file for pandas.DataFrame, produced by Step 3 Feature Extraction step

    Returns
    -------

    """
    df = load_df_pickle(df_path)

    # Filter data
    df_mask = (df["task_masks"] == True) & (df["pheno_masks"] == True) & (np.isnan(df["idpatients"]) == False)
    df_filtered = df[df_mask]

    # Construct dataframe

    data_dict = dict()
    unique_ids = np.unique(df_filtered["idpatients"])
    num_ids = unique_ids.shape[0]

    grand_arr_list, grand_tasks_list, grand_idpatients_list, grand_uniphenos_list = [], [], [], []

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
            for each_id in range(df_each.shape[0]):
                fea_arr = df_each.iloc[each_id]["features"]
                split_fea_arr = split_arr(fea_arr, stride=10)
                split_arr_list.append(split_fea_arr)

            arr_each = np.vstack(split_arr_list)
            tasks_id_arr = np.ones(arr_each.shape[0]) * task_id
            patient_id_arr = np.ones(arr_each.shape[0]) * patient_id
            uniphenos_list = [uniphenos, ] * arr_each.shape[0]

            grand_arr_list.append(arr_each)
            grand_tasks_list.append(tasks_id_arr)
            grand_idpatients_list.append(patient_id_arr)
            grand_uniphenos_list += uniphenos_list

    features = np.vstack(grand_arr_list)
    tasks = np.concatenate(grand_tasks_list)
    idpatients = np.concatenate(grand_idpatients_list)
    df_output = pd.DataFrame(
        {"features": list(features), "tasks": tasks, "idpatients": idpatients, "phenos": grand_uniphenos_list})
