from common.utils import load_df_pickle
from common.visualisation import plot2arr_skeleton
import matplotlib.pyplot as plt
import numpy as np
import skvideo.io as skv

# Checking the sanity of nan-masked skeleton


def give_arr(fig, ax):
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0.6, -0.6)

    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


df_path = "/mnt/data/raw_features_zmatrix_row_labels_withNanMasks.pickle"
df = load_df_pickle(df_path)

for i in range(5):
    print("i = {}".format(i))
    features_each_vid = df["features"][i]
    nan_masks_each_vid = df["nan_masks"][i]

    vwriter = skv.FFmpegWriter("Mov_{}.mp4".format(i))

    for t in range(128):
        fea_frame = features_each_vid[t, :, :]
        nan_mask = nan_masks_each_vid[t, :, :]
        fea_frame_masked = fea_frame * nan_mask

        # Original vid
        fig, ax = plt.subplots()
        ax.scatter(fea_frame[:, 0], fea_frame[:, 1])
        data_ori = give_arr(fig, ax)

        # masked arr
        fig2, ax2 = plt.subplots()
        ax2.scatter(fea_frame_masked[:, 0], fea_frame_masked[:, 1])
        data_masked = give_arr(fig2, ax2)

        # Concat
        draw_arr = np.concatenate([data_ori, data_masked], axis=1)
        vwriter.writeFrame(draw_arr)

    vwriter.close()
