import numpy as np
import matplotlib.pyplot as plt

# left and right is from the owner's perspective, not the observer's

videopose3d_labels = {
    0: 'hip_centre',
    1: 'r_hip', 2: 'r_knee', 3: 'r_ankle',
    4: 'l_hip', 5: 'l_knee', 6: 'l_ankle', 
    7: 'torso', 8: 'neck', 9: 'nose', 10: 'crown',
    11: 'l_shoulder', 12: 'l_elbow', 13: 'l_wrist',
    14: 'r_shoulder', 15: 'r_elbow', 16: 'r_wrist'
}

detectron_labels = {
    0: 'nose',
    1: 'l_eye', 2: 'r_eye',
    3: 'l_ear', 4: 'r_ear',
    5: 'l_shoulder', 6: 'r_shoulder',
    7: 'l_elbow', 8: 'r_elbow',
    9: 'l_wrist', 10: 'r_wrist',
    11: 'l_hip', 12: 'r_hip',
    13: 'l_knee', 14: 'r_knee',
    15: 'l_ankle', 16: 'r_ankle'

}

openpose_body25_labels = {
    0: "nose",
    1: "neck",
    2: "r_shoulder",
    3: "r_elbow",
    4: "r_wrist",
    5: "l_shoulder",
    6: "l_elbow",
    7: "l_wrist",
    8: "hip_centre",
    9: "r_hip",
    10:"r_knee",
    11:"r_ankle",
    12:"l_hip",
    13:"l_knee",
    14:"l_ankle",
    15:"r_eye",
    16:"l_eye",
    17:"r_ear",
    18:"l_ear",
    19:"l_bigtoe",
    20:"l_smalltoe",
    21:"l_heel",
    22:"r_bigtoe",
    23:"r_smalltoe",
    24:"r_heel",
}

openpose_body_draw_sequence = (
    # (0, 1, "m"),  # nose to neck
    # (0, 15, "r"),  # nose to r_eye
    # (0, 16, "l"),  # nose to l_eye
    # (15, 17, "r"),  # r_eye to r_ear
    # (16, 18, "l"),  # l_eye to l_ear
    (18, 1, "l"),  # l_ear to neck
    (17, 1, "l"),  # r_ear to neck
    (1, 5, "l"),  # neck to l_shoulder
    (5, 6, "l"),  # l_shoulder to l_elbow
    (6, 7, "l"),  # l_elbow to l_wrist
    (1, 2, "r"),  # neck to r_shoulder
    (2, 3, "r"),  # r_shoulder to r_elbow
    (3, 4, "r"),  # r_elbow to r_wrist
    (1, 8, "m"),  # neck to hip_centre
    (8, 9, "r"),  # hip_centre to r_hip
    (9, 10, "r"),  # r_hip to r_knee
    (10, 11, "r"),  # r_knee to r_ankle
    (11, 24, "r"),  # r_ankle to r_heel
    (11, 22, "r"),  # r_ankle to r_bigtoe
    (22, 23, "r"),  # r_bigtoe to r_smalltoe
    (8, 12, "l"),  # hip_centre to l_hip
    (12, 13, "l"),  # l_hip to l_knee
    (13, 14, "l"),  # l_knee to l_ankle
    (14, 21, "l"),  # l_ankle to l_heel
    (14, 19, "l"),  # l_ankle to l_bigtoe
    (19, 20, "l")  # l_bigtoe to l_small toe

)


openpose2detectron_indexes = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
openpose_L_indexes = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
openpose_R_indexes = [2, 3, 4, 9, 10, 11, 15, 17, 22, 23, 24]
openpose_central_indexes = [0, 1, 8] # From top to bottom. 0=nose, 1=neck, 8=hip_centre

excluded_points = [0, 15, 16]
excluded_points_flatten = [0, 15, 16, 25, 40, 41]

def index2feature_dist(n):
    relative_dists_indexes = np.triu_indices(25, k=1)
    x = relative_dists_indexes[0][n]
    y = relative_dists_indexes[1][n]
    return openpose_body25_labels[x], openpose_body25_labels[y]


def index2feature_asy(n):
    labels_l = [openpose_body25_labels[x] for x in openpose_L_indexes]
    labels_r = [openpose_body25_labels[x] for x in openpose_R_indexes]
    if n <11:
        l_key = labels_l[n]
        r_key = labels_r[n]
        anchor = "nose"
    elif n >=11 and n < 22:
        n_corrected = n-11
        l_key = labels_l[n_corrected]
        r_key = labels_r[n_corrected]
        anchor = "neck"
    elif n >=22:
        n_corrected = n-22
        l_key = labels_l[n_corrected]
        r_key = labels_r[n_corrected]
        anchor = "hip_centre"
    return l_key, r_key, anchor
    
def index2feature(n):
    x, y, anchor = None, None, None
    if n <300:
        x, y = index2feature_dist(n)
        
    elif n >= 300 and n < 333:
        n_corrected = n -300
        x, y, anchor = index2feature_asy(n_corrected)
        
    elif n >= 333 and n < 633:
        n_corrected = n-333
        x, y = index2feature_dist(n_corrected)
        
    elif n >= 633:
        n_corrected = n-633
        x, y, anchor = index2feature_asy(n_corrected)
    
    if anchor is None:
        feature = "-(DIST) {}-{}".format(x,y)
        
    else:
        feature = "*(ASY) {}-{}".format(x[2:], anchor)
    
    return feature


def draw_skeleton(ax, x, y):
    side_dict = {
        "m": "k",
        "l": "r",
        "r": "b"
    }
    for start, end, side in openpose_body_draw_sequence:
        ax.plot(x[[start, end]], y[[start, end]], c=side_dict[side])
    return ax


def plot2arr_skeleton(x, y, title, x_lim=(-0.6, 0.6), y_lim=(0.6, -0.6)):
    fig, ax = plt.subplots()
    ax.scatter(np.delete(x, excluded_points), np.delete(y, excluded_points))
    ax = draw_skeleton(ax, x, y)
    fig.suptitle(title)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

