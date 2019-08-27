from common.utils import load_df_pickle, idx2task, task2idx, idx2pheno, pheno2idx
from common.visualisation import draw_skeleton_custom
from common.keypoints_format import excluded_points
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


df_path = "/mnt/thesis_results/data/model_outputs_full.pickle"
df_phenos_path = "/mnt/thesis_results/data/model_phenos_outputs_full.pickle"
df = load_df_pickle(df_path)
df_phenos = load_df_pickle(df_phenos_path)
df["std"] =  list(-np.log(np.mean(np.std(np.stack(list(df["ori_motion"])), axis=2), axis=1)))
print(df.columns)
model_names = ["B", "B+C", "B+C+T", "B+C+T+P"]


class MotionDrawer:
    def __init__(self, df, motion_cols, name_cols, save_img_path, x_sep, y_sep, interval=[0, 127, 1], figsize=(16, 12),
                 scatter_size=5, dpi=300):

        # Copy arguments
        self.motion_cols = motion_cols
        self.name_cols = name_cols
        self.save_img_path = save_img_path
        self.interval = interval

        # Data frames/calculation related
        self.num_models = len(self.motion_cols)
        self.df_motions = df[self.motion_cols]
        self.time_indexes = np.arange(*interval)
        self.x_translation = np.arange(self.time_indexes.shape[0]) * x_sep
        self.y_translation = np.arange(self.num_models) * y_sep
        self.margin = 0.2
        self.text_space = 0.5
        self.num_times = self.time_indexes.shape[0]

        # Matplotlib attributes
        self.alpha_offset = 0.3
        self.figsize = figsize
        self.scatter_size = scatter_size
        self.dpi = dpi
        # Reserved
        self.x, self.y = None, None
        self.x_bound, self.y_bound = None, None

    def draw(self, i):
        # Select and translate skeletons.
        self._convert_selected_data_to_arr(i)
        self._translate_skeletons()
        self._get_plot_boundary()  # Pre-calculate plot window boundarys

        # Plot the motions
        ax = self._plot_motions()

        # Plot's setting
        ax.set_xlim(*self.x_bound)
        ax.set_ylim(*self.y_bound[::-1])
        ax.axis("off")

        # Save figure
        if self.save_img_path:
            plt.savefig(self.save_img_path, dpi=self.dpi)
        return ax

    def _convert_selected_data_to_arr(self, i):
        """
        1. Select i index from dataframe.
        2. Stack each (50, 128) motion to (num_models, 50, 128)
        3. Slice (num_models, 50, 128) to (num_models, 50, num_times), as defined by the selected interval

        """
        motions_list = []
        for motion_col in self.motion_cols:
            motions_list.append(self.df_motions[motion_col].iloc[i])
        motion_arrs = np.stack(motions_list)
        self.x, self.y = motion_arrs[:, 0:25, self.time_indexes], motion_arrs[:, 25:, self.time_indexes]

    def _translate_skeletons(self):
        self.y = self.y + self.y_translation.reshape(self.y_translation.shape[0], 1, 1)
        self.x = self.x + self.x_translation.reshape(1, 1, self.x_translation.shape[0])

    def _get_plot_boundary(self):
        x_min, x_max = np.min(self.x) - self.text_space, np.max(self.x) + self.margin
        y_min, y_max = np.min(self.y), np.max(self.y)
        self.x_bound, self.y_bound = (x_min, x_max), (y_min, y_max)

    def _plot_motions(self):
        fig, ax = plt.subplots(figsize=self.figsize)
        for m in range(self.num_models):
            for t in range(self.num_times):
                time_portion = (t + 1) / self.num_times * (
                            1 - self.alpha_offset)  # scale the time_portion s.t. time_portion \in [0, 1-1-self.alpha_offset]
                x_to_plot, y_to_plot = self.x[m, :, t], self.y[m, :, t]
                ax = draw_skeleton_custom(ax, x=x_to_plot, y=y_to_plot, c="k", alpha=self.alpha_offset + time_portion)
                ax.scatter(x=np.delete(x_to_plot, excluded_points), y=np.delete(y_to_plot, excluded_points), c="red",
                           s=self.scatter_size, alpha=self.alpha_offset + time_portion)
            ax.text(self.x_bound[0] + 0.1, self.y_translation[m], "{}".format(self.name_cols[m]))
        return ax

plt.ioff()

for img_idx in range(df.shape[0]):
    print("\rimage: %d" %img_idx, flush=True, end="")
    selected_idx = img_idx
    selected_interval = [0, 128, 10]
    motion_cols = ["ori_motion", "B_recon", "B+C_recon", "B+C+T_recon", "B+C+T+P_recon"]
    name_cols = ["Original", "B", "B+C", "B+C+T", "B+C+T+P"]
    folder_dir = "/mnt/thesis_results/recon_examples_full"
    os.makedirs(folder_dir, exist_ok=True)
    save_img_path = os.path.join(folder_dir, "{}_{}.png".format(selected_idx, str(selected_interval)))
#     save_img_path=None
    drawer = MotionDrawer(df, motion_cols, name_cols, save_img_path, x_sep=0.4, y_sep=1, interval=selected_interval, scatter_size=5, dpi=100)
    ax = drawer.draw(selected_idx)
    plt.close()