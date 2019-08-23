from common.utils import tensor2numpy, write_df_pickle
import pandas as pd
import numpy as np
import umap

class OutputSavers:
    """
    Things to save:
    From inputs:
        1. Original motion
        2. Labels of tasks
        3. Labels of Phenotypes

    From B:
        1. Reconstructed motion
        2. Latents
    From B+C:
        1. Reconstructed motion
        2. Latents
    From B+C+T
        1. Reconstructed motion
        2. Latents
        3. Task prediction
    From B+C+T+P
        1. Reconstructed motion
        2. Latents
        3. Task prediction
        4. Phenotype prediction

    """

    def __init__(self, data_gen, model_container_set, identifier_set, save_df_path, save_pheno_df_path):
        """
        model_container_set : list
            Predefined as models of ["B", "B+C", "B+C+T", "B+C+T+P"]
        identifier_set : list
            Predefined as ["B", "B+C", "B+C+T", "B+C+T+P"]
        """
        self.data_gen = data_gen
        self.data_gen.mt = data_gen.df_test.shape[0]
        self.identifier_set = [x.replace("Thesis_", "") for x in identifier_set]
        for i in range(len(identifier_set)):
            del model_container_set[i].data_gen
        self.model_container_set = model_container_set
        self.df_dict = dict()
        self.df_pheno_dict = dict()
        self.save_df_path = save_df_path
        self.save_pheno_df_path = save_pheno_df_path


    def forward_batch(self):

        for _, test_data in self.data_gen.iterator():
            pass

        x, nan_masks, tasks_np, tasks_mask_np, phenos_np, phenos_mask_np, towards, _, _, idpatients_np = test_data

        self.df_dict["ori_motion"] = list(x)
        self.df_dict["ori_motion_mask"] = list(nan_masks)
        self.df_dict["task"] = list(tasks_np)
        self.df_dict["task_mask"] = list(tasks_mask_np)
        self.df_dict["pheno"] = list(phenos_np)
        self.df_dict["pheno_mask"] = list(phenos_mask_np)
        self.df_dict["direction"] = list(towards)


        for identifier, model_container in zip(self.identifier_set, self.model_container_set):
            print("forward passing {}".format(identifier))
            data_outputs = model_container.forward_evaluate(test_data)
            if identifier == "B+C+T+P":
                recon, pred_task, _, motion_info, phenos_info = data_outputs
                phenos_pred, phenos_labels_np = phenos_info
            else:
                recon, pred_task, _, motion_info = data_outputs
                phenos_pred, phenos_labels_np = None, None
            motion_z, _, _ = motion_info

            data_to_record = (recon, motion_z, pred_task, phenos_pred, phenos_labels_np)

            self._record_data_by_identifier(identifier, data_to_record)
        self._save_dfs()

    def _record_data_by_identifier(self, identifier, data_to_record):

        (recon, motion_z, pred_task, phenos_pred, phenos_labels_np) = data_to_record
        recon, motion_z, pred_task = tensor2numpy(recon, motion_z, pred_task)

        if identifier == "B+C+T":
            self.df_dict["{}_pred_task".format(identifier)] = list(np.argmax(pred_task, axis=1))

        elif identifier == "B+C+T+P":
            phenos_pred = tensor2numpy(phenos_pred)[0]
            self.df_dict["{}_pred_task".format(identifier)] = list(np.argmax(pred_task, axis=1))
            self.df_pheno_dict["pheno_pred"] = list(np.argmax(phenos_pred, axis=1))
            self.df_pheno_dict["pheno_labels"] = list(phenos_labels_np)
        self.df_dict["{}_recon".format(identifier)] = list(recon)
        self.df_dict["{}_z".format(identifier)] = list(motion_z)
        self.df_dict["{}_z_umap".format(identifier)] = list(self._umap_transform(motion_z))

    def _save_dfs(self):
        df = pd.DataFrame(self.df_dict)
        df_phenos = pd.DataFrame(self.df_pheno_dict)
        write_df_pickle(df, self.save_df_path)
        write_df_pickle(df_phenos, self.save_pheno_df_path)

    def _umap_transform(self, motion_z):
        umapper = umap.UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.1,
            metric="euclidean"
        )
        z_umap = umapper.fit_transform(motion_z)
        return z_umap
