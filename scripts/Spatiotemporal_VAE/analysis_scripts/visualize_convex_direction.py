from .basic_visualize_latent_space import concat_generator_batches
from common.utils import expand1darr, numpy2tensor, tensor2numpy, write_df_pickle
from common.visualisation import LatentSpaceVideoVisualizer, SkeletonPainter
import torch
import os
import pandas as pd
import numpy as np


def convert_direction_convex_combinaiton(labels, frac):
    onehot2d = expand1darr(labels, 3)
    one_mask = onehot2d[:, 1, 0] == 1
    two_mask = onehot2d[:, 2, 0] == 1
    onehot2d[one_mask, 1, :] = onehot2d[one_mask, 1, :] * frac
    onehot2d[one_mask, 2, :] = 1 - onehot2d[one_mask, 1, :]
    onehot2d[two_mask, 2, :] = onehot2d[two_mask, 2, :] * frac
    onehot2d[two_mask, 1, :] = 1 - onehot2d[two_mask, 2, :]
    return onehot2d


def save_for_interplot_direction_fraction(x, recon_list, fraction_list, motion_z_umap, pheno_labels, tasks_labels,
                                          towards_labels,
                                          save_data_dir, df_save_fn, vid_dirname, draw=False):
    # Define and make directories
    compare_vids_dir = os.path.join(save_data_dir, "videos", vid_dirname)
    os.makedirs(compare_vids_dir, exist_ok=True)

    # Produce dictionary for recon
    recon_dict = {"recon_%0.2f" % (fraction): list(recon) for recon, fraction in zip(recon_list, fraction_list)}

    # Save arrays
    df = pd.DataFrame({"ori_motion": list(x),
                       "motion_z_umap": list(motion_z_umap),
                       "phenotype": list(pheno_labels),
                       "task": list(tasks_labels),
                       "direction": list(towards_labels)
                       })
    df_recon = pd.DataFrame(recon_dict)
    df_concat = pd.concat((df, df_recon), axis=1)
    write_df_pickle(df_concat, os.path.join(save_data_dir, df_save_fn))

    return


def draw_videos_direction_fraction(x, recon_list, fracion_list, save_data_dir, vid_dirname):
    num_vid = x.shape[0]





def save_for_convex_direction(model_container, data_gen, fit_samples_num, vis_data_dir, model_identifier):
    # Get data with (relatively) equal sizes of phenotypes, and base data for fitting
    equal_pheno_info, base_info = concat_generator_batches(data_gen=data_gen, fit_samples_num=fit_samples_num)
    (x_equal_pheno, tasks_equal_pheno, phenos_equal_pheno, towards_equal_pheno) = equal_pheno_info
    (x_base, tasks_base, phenos_base, towards_base) = base_info

    # Forward pass for base
    x_equal_pheno, x_base = numpy2tensor(model_container.device, x_equal_pheno, x_base)
    towards2d_base = convert_direction_convex_combinaiton(towards_equal_pheno, 1)
    recon_motion_base, pose_z_seq_base, recon_pose_z_seq_base, motion_z_base = model_container._forward_pass(x_base,
                                                                                                             towards2d_base)

    # Fit Umap with base
    vis = LatentSpaceVideoVisualizer(model_identifier=model_identifier, save_vid_dir=None)
    vis.fit_umap(pose_z_seq=pose_z_seq_base, motion_z=motion_z_base)

    # Forward pass for each fraction
    fraction_list = [1, 0.75, 0.5, 0.25, 0]
    recon_list = []

    # -Encode to obtain latents
    with torch.no_grad():
        towards2d_equal = convert_direction_convex_combinaiton(towards_equal_pheno, 1)
        (pose_z_seq_equal, _, _), (motion_z_equal, _, _) = model_container.model.encode(x_equal_pheno, towards2d_equal)

    # -Decoding with respect to each fraction
    for direction_frac in fraction_list:
        towards2d_equal = convert_direction_convex_combinaiton(towards_equal_pheno, direction_frac)
        recon_motion_equal, recon_pose_z_seq_equal, _ = model_container.model.decode(motion_z_equal, towards2d_equal)
        recon_motion_equal = tensor2numpy(recon_motion_equal)[0]
        recon_list.append(recon_motion_equal)

    # Project on umap
    x_equal_pheno, motion_z_equal = tensor2numpy(x_equal_pheno, motion_z_equal)
    motion_z_equal_umap = vis.motion_z_umapper.transform(motion_z_equal)

    # Visualization
    save_for_interplot_direction_fraction(x=x_equal_pheno,
                                          recon_list=recon_list,
                                          fraction_list=fraction_list,
                                          motion_z_umap=motion_z_equal_umap,
                                          pheno_labels=phenos_equal_pheno,
                                          tasks_labels=tasks_equal_pheno,
                                          towards_labels=towards_equal_pheno,
                                          save_data_dir=vis_data_dir,
                                          df_save_fn="direction_fraction_CN_K-0.0001.pickle",
                                          vid_dirname="direction_fraction")
