from spatiotemporal_vae_script import load_model_container
from Spatiotemporal_VAE.STVAE_run import CSTVAEmodel
import torch
import numpy as np
import pandas as pd
from common.utils import numpy2tensor, expand1darr, tensor2numpy, write_df_pickle, load_df_pickle
from common.generator import GaitGeneratorFromDFforTemporalVAE

df = load_df_pickle("data_frame.pickle")
import pdb
pdb.set_trace()

df_path = "/mnt/data/feas_tasks_phenos_nanMasks_idpatient.pickle"
model_identifier = "CB-K(0.0001)-C-G-S2-New"  # Only Direction


class test_generator(GaitGeneratorFromDFforTemporalVAE):
    def _convert_df_to_data(self, df_shuffled, start, stop):
        selected_df = df_shuffled.iloc[start:stop, :].copy()

        # Retrieve train data
        x_train_info, task_train_info, pheno_train_info, towards_train = self._loop_for_array_construction(
            selected_df,
            self.m)
        x_train, x_train_masks = x_train_info
        task_train, task_train_masks = task_train_info
        pheno_train, pheno_train_masks = pheno_train_info

        # Retrieve test data
        x_test_info, task_test_info, pheno_test_info, towards_test = self._loop_for_array_construction(
            self.df_test,
            self.df_test.shape[0])

        x_test, x_test_masks = x_test_info
        task_test, task_test_masks = task_test_info
        pheno_test, pheno_test_masks = pheno_test_info

        # Combine as output
        train_info = (x_train, x_train_masks, task_train, task_train_masks, pheno_train, pheno_train_masks,
                      towards_train)
        test_info = (x_test, x_test_masks, task_test, task_test_masks, pheno_test, pheno_test_masks,
                     towards_test)

        return train_info, test_info

    def _loop_for_array_construction(self, df, num_samples):
        # fea_vec/fea_mask_vec ~ (num_frames, 25, 2), task ~ int, task_mask ~ bool (True for non-nan, False for nan)
        # pheno ~ int, pheno_mask ~ bool (True for non-nan, False for nan), towards ~ int (0=unknown, 1=left, 2=right)
        # leg ~ float, leg_mask ~ bool (True for non-nan, False for nan)
        select_list = ["features", "feature_masks", "tasks", "task_masks", "phenos", "pheno_masks",
                       "towards_camera"]

        df_np = np.asarray(df[select_list].iloc[0:num_samples])

        fea_vec, fea_mask_vec, task, task_mask, pheno, pheno_mask, towards = list(df_np.T)

        task, task_mask = task.astype(np.int), task_mask.astype(np.bool)
        pheno, pheno_mask = pheno.astype(np.int), pheno_mask.astype(np.bool)
        towards

        features_arr = np.zeros((num_samples, self.total_fea_dims, self.n))
        fea_masks_arr = np.zeros(features_arr.shape)

        for i in range(num_samples):
            # Slice to the receptive window
            slice_start = np.random.choice(fea_vec[i,].shape[0] - self.n)
            fea_vec_sliced = fea_vec[i][slice_start:slice_start + self.n, :, :]
            fea_mask_vec_sliced = fea_mask_vec[i][slice_start:slice_start + self.n, :, :]

            # Construct output
            x_end_idx, y_end_idx = self.keyps_x_dims, self.keyps_x_dims + self.keyps_y_dims
            features_arr[i, 0:x_end_idx, :] = fea_vec_sliced[:, :, 0].T  # Store x-coordinates
            features_arr[i, x_end_idx:y_end_idx, :] = fea_vec_sliced[:, :, 1].T  # Store y-coordinates
            fea_masks_arr[i, 0:x_end_idx, :] = fea_mask_vec_sliced[:, :, 0].T  # Store x-coordinates
            fea_masks_arr[i, x_end_idx:y_end_idx, :] = fea_mask_vec_sliced[:, :, 1].T  # Store y-coordinates
        return (features_arr, fea_masks_arr), (task, task_mask), (pheno, pheno_mask), towards


class test_model(CSTVAEmodel):

    def _convert_input_data(self, train_data, test_data):
        x, nan_masks, tasks, tasks_mask, phenos, phenos_mask, towards = train_data
        x_test, nan_masks_test, tasks_test, tasks_mask_test, phenos_test, phenos_mask_test, towards_test = test_data

        # Convert numpy to torch.tensor
        x, x_test = numpy2tensor(self.device, x, x_test)
        tasks = torch.from_numpy(tasks).long().to(self.device)
        tasks_test = torch.from_numpy(tasks_test).long().to(self.device)
        tasks_mask = torch.from_numpy(tasks_mask * 1).float().to(self.device)
        tasks_mask_test = torch.from_numpy(tasks_mask_test * 1).float().to(self.device)
        nan_masks = torch.from_numpy(nan_masks * 1).float().to(self.device)
        nan_masks_test = torch.from_numpy(nan_masks_test * 1).float().to(self.device)
        towards, towards_test = numpy2tensor(self.device,
                                             expand1darr(towards.astype(np.int64), 3, self.seq_dim),
                                             expand1darr(towards_test.astype(np.int64), 3, self.seq_dim)
                                             )

        train_inputs = (x, towards)
        test_inputs = (x_test, towards_test)
        train_info = (x, nan_masks, tasks, tasks_mask, phenos, phenos_mask, towards)
        test_info = (x_test, nan_masks_test, tasks_test, tasks_mask_test, phenos_test, phenos_mask_test, towards_test)
        return train_inputs, test_inputs, train_info, test_info


model_container, save_model_path = load_model_container(model_class=test_model,
                                                        model_identifier=model_identifier,
                                                        df_path=df_path,
                                                        datagen_batch_size=512)

data_gen = test_generator(df_path, m=512)
model_container.data_gen = data_gen

tasks_train_list, tasks_mask_train_list, tasks_test_list, tasks_mask_test_list = [], [], [], []
phenos_train_list, phenos_mask_train_list, phenos_test_list, phenos_mask_test_list = [], [], [], []
input_train_list, input_mask_train_list, input_test_list, input_mask_test_list = [], [], [], []
recon_train_list, recon_test_list = [], []
latent_train_list, latent_test_list = [], []

iter = 0
for train_data, test_data in model_container.data_gen.iterator():
    print(iter)
    iter += 1

    train_inputs, test_inputs, train_converted, test_converted = model_container._convert_input_data(train_data,
                                                                                                     test_data)
    x, nan_masks, labels, labels_mask, phenos, phenos_mask, cond = train_converted
    x_test, nan_masks_test, labels_test, labels_mask_test, phenos_test, phenos_test_mask, cond_test = test_converted

    recon_train, _, _, latent_train = model_container._forward_pass(*train_inputs)
    recon_test, _, _, latent_test = model_container._forward_pass(*test_inputs)

    x, nan_masks, recon_train, latent_train, labels, labels_mask = tensor2numpy(
        x,
        nan_masks,
        recon_train,
        latent_train,
        labels,
        labels_mask
    )

    tasks_train_list.append(labels)
    tasks_mask_train_list.append(labels_mask)
    phenos_train_list.append(phenos)
    phenos_mask_train_list.append(phenos_mask)
    input_train_list.append(x)
    input_mask_train_list.append(nan_masks)
    recon_train_list.append(recon_train)
    latent_train_list.append(latent_train)

x_test, nan_masks_test, recon_test, latent_test, labels_test, labels_mask_test = tensor2numpy(
    x_test,
    nan_masks_test,
    recon_test,
    latent_test,
    labels_test,
    labels_mask_test
)
tasks_train_np = np.concatenate(tasks_train_list, axis=0)
tasks_mask_train_np = np.concatenate(tasks_mask_train_list, axis=0)
phenos_train_np = np.concatenate(phenos_train_list, axis=0)
phenos_mask_train_np = np.concatenate(phenos_mask_train_list, axis=0)
input_train_np = np.concatenate(input_train_list, axis=0)
input_mask_train_np = np.concatenate(input_mask_train_list, axis=0)
recon_train_np = np.concatenate(recon_train_list, axis=0)
latent_train_np = np.concatenate(latent_train_list, axis=0)

train_flag = np.ones(tasks_train_np.shape[0])
test_flag = np.zeros(x_test.shape[0])

tasks_df = np.concatenate([tasks_train_np, labels_test], axis=0)
tasks_mask_df = np.concatenate([tasks_mask_train_np, labels_mask_test], axis=0)
phenos_df = np.concatenate([phenos_train_np, phenos_test], axis=0)
phenos_mask_df = np.concatenate([phenos_mask_train_np, phenos_test_mask], axis=0)
input_df = np.vstack([input_train_np, x_test])
input_mask_df = np.vstack([input_mask_train_np, nan_masks_test])
recon_df = np.vstack([recon_train_np, recon_test])
latent_df = np.vstack([latent_train_np, latent_test])
train_flag = np.concatenate([train_flag, test_flag], axis=0)
df = pd.DataFrame({
    "input": list(input_df),
    "input_mask": list(input_mask_df),
    "recon": list(recon_df),
    "latent": list(latent_df),
    "tasks": list(tasks_df),
    "tasks_mask": list(tasks_mask_df),
    "phenos": list(phenos_df),
    "phenos_mask": list(phenos_mask_df),
    "train_flag": list(train_flag)
})


write_df_pickle(df, "data_frame.pickle")
