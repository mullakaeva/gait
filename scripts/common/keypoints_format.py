import numpy as np

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



openpose2detectron_indexes = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
openpose_L_indexes = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
openpose_R_indexes = [2, 3, 4, 9, 10, 11, 15, 17, 22, 23, 24]
openpose_central_indexes = [0, 1, 8] # From top to bottom. 0=nose, 1=neck, 8=hip_centre


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
