import numpy as np
import pandas as pd
import os
import umap
from common.data_preparation import prepare_data_for_concatenated_latent
from common.utils import numpy2tensor, tensor2numpy, write_df_pickle


class FingerprintSaver:
    def __init__(self, model_container, load_df_path, save_data_dir, save_df_name):
        print("Constructing dataframe of each patient's fingerprint.")
        self.model_container = model_container
        self.df = self._load_dataframe(load_df_path)
        self.save_data_dir = save_data_dir
        self.save_df_name = save_df_name

        # Reserve
        self.task_means = dict()
        self.df_concat = None

    def process_and_save(self):
        # Forward pass and add column
        self._forward_pass()

        # Calculate grand mean for each task across all patients
        self._calc_grand_mean_each_task_all_patients()

        # Calculate the 8 task's means for each patient
        self.df_concat = self._calc_mean_each_task_each_patient()

        # Umap embedding
        self._umap_fit_and_project()

        # Save the dataframe
        self._save_dataframe()

    def _load_dataframe(self, df_path):
        # Load data
        df_processed = prepare_data_for_concatenated_latent(df_path, equal_phenos=False)
        df_shuffled = df_processed.sample(frac=1, random_state=60).reset_index(drop=True)
        del df_processed
        return df_shuffled

    def _forward_pass(self):
        x = np.asarray(list(self.df["features"]))

        motion_z_list = []
        batch = 512
        batch_times = int(x.shape[0] / batch)
        for i in range(batch_times + 1):
            if i < batch_times:
                x_each = numpy2tensor(self.model_container.device, x[i * batch: (i + 1) * batch, ])[0]
            else:
                x_each = numpy2tensor(self.model_container.device, x[i * batch:, ])[0]
            _, _, _, motion_z_batch = self.model_container._forward_pass(x_each)
            motion_z_batch = tensor2numpy(motion_z_batch)[0]
            motion_z_list.append(motion_z_batch)

        motion_z = np.vstack(motion_z_list)
        self.df["motion_z"] = list(motion_z)
        # Averaging over split video rows and delete unnecessary columns
        del self.df["features"]  # The "features" column is not needed anymore after forward pass
        self.df = self.df.groupby("avg_idx", as_index=False).apply(
            np.mean)  # Average across avg_idx. They are split from the same video.
        del self.df["avg_idx"]

    def _calc_grand_mean_each_task_all_patients(self):
        print("Calculating grand means")
        for task_idx in range(8):
            mask = self.df["tasks"] == task_idx
            self.task_means[task_idx] = np.mean(np.asarray(list(self.df[mask].motion_z)), axis=0)

    def _calc_mean_each_task_each_patient(self):
        # Calc means for each tasks in each patients
        print("Calculating patient's task's mean")
        all_patient_ids = np.unique(self.df["idpatients"])
        num_patient_ids = all_patient_ids.shape[0]
        patient_id_list, features_list, phenos_list = [], [], []

        for p_idx in range(num_patient_ids):
            print("\rpatient {}/{}".format(p_idx, num_patient_ids), flush=True, end="")
            patient_id = all_patient_ids[p_idx]
            patient_mask = self.df["idpatients"] == patient_id
            unique_tasks = np.unique(self.df[patient_mask]["tasks"])
            unique_phenos = np.unique(np.concatenate(list(self.df[patient_mask]["phenos"])))
            task_vec_list = []

            for task_idx in range(8):

                if task_idx not in unique_tasks:
                    task_vec_list.append(self.task_means[task_idx])
                else:
                    mask = (self.df["idpatients"] == patient_id) & (self.df["tasks"] == task_idx)
                    patient_task_mean = np.mean(np.asarray(list(self.df[mask]["motion_z"])), axis=0)
                    task_vec_list.append(patient_task_mean)
            task_vec = np.concatenate(task_vec_list)
            patient_id_list.append(patient_id)
            features_list.append(task_vec)
            phenos_list.append(unique_phenos)

        df_concat = pd.DataFrame({"patient_id": patient_id_list,
                                  "fingerprint": features_list,
                                  "phenos": phenos_list})
        return df_concat

    def _umap_fit_and_project(self):
        print("Umapping")
        fingerprint_umapper = umap.UMAP(n_neighbors=15,
                                        n_components=2,
                                        min_dist=0.1,
                                        metric="euclidean")
        fingerprint_z = fingerprint_umapper.fit_transform(np.asarray(list(self.df_concat["fingerprint"])))
        self.df_concat["fingerprint_z"] = list(fingerprint_z)

    def _save_dataframe(self):
        write_df_pickle(self.df_concat, os.path.join(self.save_data_dir,
                                                     self.save_df_name))

class CondFingerprintSaver(FingerprintSaver):

    def _forward_pass(self):

        x = np.asarray(list(self.df["features"]))
        directions = np.asarray(list(self.df["directions"]))

        motion_z_list = []
        batch = 512
        batch_times = int(x.shape[0] / batch)
        for i in range(batch_times + 1):
            if i < batch_times:
                direction_each = directions[i * batch: (i + 1) * batch, ]
                x_each, direction_each = numpy2tensor(self.model_container.device, x[i * batch: (i + 1) * batch, ],
                                                      direction_each)
            else:
                direction_each = directions[i * batch:, ]
                x_each, direction_each = numpy2tensor(self.model_container.device, x[i * batch:, ], direction_each)

            _, _, _, motion_z_batch = self.model_container._forward_pass(x_each, direction_each)
            motion_z_batch = tensor2numpy(motion_z_batch)[0]
            motion_z_list.append(motion_z_batch)

        motion_z = np.vstack(motion_z_list)
        self.df["motion_z"] = list(motion_z)
        # Averaging over split video rows and delete unnecessary columns
        del self.df["features"]  # The "features" column is not needed anymore after forward pass
        self.df = self.df.groupby("avg_idx", as_index=False).apply(
            np.mean)  # Average across avg_idx. They are split from the same video.
        del self.df["avg_idx"]
