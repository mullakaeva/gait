import numpy as np
import os
import pandas as pd
from common.utils import numpy2tensor, tensor2numpy, expand1darr, write_df_pickle
from common.visualisation import LatentSpaceVideoVisualizer, gen_single_vid_two_skeleton_motion


def concat_generator_batches(data_gen, fit_samples_num):
    # Lists for concatenation
    x_equal_phenos_list, tasks_equal_list, phenos_equal_list, towards_equal_list = [], [], [], []
    # Get data from data generator's first loop
    for train_data, test_data, towards_info in data_gen.iterator():

        # x_fit for umap embedding
        x, x_masks, tasks, task_masks, phenos, pheno_masks = train_data
        towards, _ = towards_info
        masks = (task_masks == 1) & (pheno_masks == 1)
        x, tasks, phenos, towards = x[masks,].copy(), tasks[masks,], phenos[masks,], towards[masks,]

        # Produce phenos of equal/similar amounts
        uniphenos, phenos_counts = np.unique(phenos, return_counts=True)
        max_counts = np.sort(phenos_counts)[3]

        # Clap the maximum count of phenotype labels, s.t. certain label won't overrepresent the visualization
        for pheno_idx in range(13):
            x_each_pheno = x[phenos == pheno_idx,]
            tasks_each_pheno = tasks[phenos == pheno_idx,]
            phenos_each_pheno = phenos[phenos == pheno_idx,]
            towards_each_pheno = towards[phenos == pheno_idx,]
            x_equal_phenos_list.append(x_each_pheno[0:max_counts, ])
            tasks_equal_list.append(tasks_each_pheno[0:max_counts, ])
            phenos_equal_list.append(phenos_each_pheno[0:max_counts, ])
            towards_equal_list.append(towards_each_pheno[0:max_counts, ])

        # Concatenate and prepare data
        x_equal_pheno = np.vstack(x_equal_phenos_list)
        tasks_equal_pheno = np.concatenate(tasks_equal_list)
        phenos_equal_pheno = np.concatenate(phenos_equal_list)
        towards_equal_pheno = np.concatenate(towards_equal_list)

        np.random.seed(50)
        ran_vec = np.random.permutation(x_equal_pheno.shape[0])
        x_equal_pheno, tasks_equal_pheno, phenos_equal_pheno, towards_equal_pheno = x_equal_pheno[ran_vec,], \
                                                                                    tasks_equal_pheno[ran_vec,], \
                                                                                    phenos_equal_pheno[ran_vec,], \
                                                                                    towards_equal_pheno[ran_vec]

        x_base, tasks_base, phenos_base, towards_base = x[0:fit_samples_num, ], tasks[0:fit_samples_num, ], \
                                                        phenos[0:fit_samples_num, ], towards[0:fit_samples_num, ]

        equal_pheno_info = (x_equal_pheno, tasks_equal_pheno, phenos_equal_pheno, towards_equal_pheno)
        base_info = (x_base, tasks_base, phenos_base, towards_base)

    return equal_pheno_info, base_info

def convert_towards2onehot(towards, model_container):
    expanded_towards = expand1darr(towards.astype(np.int64),
                                   model_container.conditional_label_dim,
                                   model_container.seq_dim)

    return expanded_towards

def save_vis_data_for_interactiveplot(x, recon, motion_z_umap, pheno_labels, tasks_labels, towards_labels,
                                      save_data_dir, df_save_fn, vid_dirname, draw=False):
    # Define and make directories
    compare_vids_dir = os.path.join(save_data_dir, "videos", vid_dirname)
    os.makedirs(compare_vids_dir, exist_ok=True)

    # Save arrays
    df = pd.DataFrame({"ori_motion": list(x),
                       "recon_motion": list(recon),
                       "motion_z_umap": list(motion_z_umap),
                       "phenotype": list(pheno_labels),
                       "task": list(tasks_labels),
                       "direction": list(towards_labels)
                       })
    write_df_pickle(df, os.path.join(save_data_dir, df_save_fn))

    if draw:
        # Draw videos
        total_vids_num = x.shape[0]
        for i in range(total_vids_num):
            print("\rWriting {}/{} input and recon videos".format(i, total_vids_num), end="", flush=True)
            compare_vid_save_path = os.path.join(compare_vids_dir, "%s_%d.mp4" % (vid_dirname, i))
            gen_single_vid_two_skeleton_motion(x[i,], recon[i,], compare_vid_save_path)



def save_for_latent_vis(model_container, data_gen, fit_samples_num, vis_data_dir, model_identifier):
    # Get data with (relatively) equal sizes of phenotypes, and base data for fitting
    equal_pheno_info, base_info = concat_generator_batches(data_gen=data_gen, fit_samples_num=fit_samples_num)
    (x_equal_pheno, tasks_equal_pheno, phenos_equal_pheno, towards_equal_pheno) = equal_pheno_info
    (x_base, tasks_base, phenos_base, towards_base) = base_info

    # Forward pass
    towards_equal_pheno = convert_towards2onehot(towards_equal_pheno, model_container)
    towards_base = convert_towards2onehot(towards_base, model_container)
    x_equal_pheno, x_base = numpy2tensor(model_container.device, x_equal_pheno, x_base)
    recon_motion_equal, pose_z_seq_equal, recon_pose_z_seq_equal, motion_z_equal = model_container._forward_pass(
        x_equal_pheno,
        towards_equal_pheno)
    recon_motion_base, pose_z_seq_base, recon_pose_z_seq_base, motion_z_base = model_container._forward_pass(x_base,
                                                                                                             towards_base)

    # Fit Umap embedding
    vis = LatentSpaceVideoVisualizer(model_identifier=model_identifier, save_vid_dir=None)
    vis.fit_umap(pose_z_seq=pose_z_seq_base, motion_z=motion_z_base)
    x_equal_pheno, recon_motion_equal, motion_z_equal = tensor2numpy(x_equal_pheno, recon_motion_equal,
                                                                     motion_z_equal)
    motion_z_equal_umap = vis.motion_z_umapper.transform(motion_z_equal)

    save_vis_data_for_interactiveplot(x=x_equal_pheno,
                                      recon=recon_motion_equal,
                                      motion_z_umap=motion_z_equal_umap,
                                      pheno_labels=phenos_equal_pheno,
                                      tasks_labels=tasks_equal_pheno,
                                      towards_labels=towards_equal_pheno,
                                      save_data_dir=vis_data_dir,
                                      df_save_fn="conditional-0.0001-latent_space.pickle",
                                      vid_dirname="equal_phenos")
    return
