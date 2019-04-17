from common.generator import GaitGeneratorFromDF
import matplotlib.pyplot as plt
df_path = "/mnt/data/raw_features_zmatrix_row_labels.pickle"

gen = GaitGeneratorFromDF(df_path)
m = 15
for sample in gen.iterator():
    for i in range(sample.shape[1]):
        if i % 50 == 0:
            sample_keyps = sample.reshape(gen.m, gen.n, 25, 3)
            plt.scatter(sample_keyps[m, i, :, 0], sample_keyps[m, i, :, 1])
            plt.savefig("check_{}.png".format(i))
    break

