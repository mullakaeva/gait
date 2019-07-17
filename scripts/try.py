from common.generator import GaitGeneratorFromDFforTemporalVAE
from common.utils import expand1darr
from Spatiotemporal_VAE import ConditionalModel
import numpy as np
if __name__ == "__main__":
    df_path = "/mnt/data/feas_tasks_phenos_nanMasks_idpatient.pickle"
    data_gen = GaitGeneratorFromDFforTemporalVAE(df_pickle_path=df_path, m=32, n=128, seed=60)

    for train_info, test_info, towards_info in data_gen.iterator():
        (x_train, x_train_masks, task_train, task_train_masks, pheno_train, pheno_train_masks) = train_info
        (x_test, x_test_masks, task_test, task_test_masks, pheno_test, pheno_test_masks) = test_info
        (towards_train, towards_test) = towards_info



        towards_train


    vec = np.arange(3)
    print(vec)

    vec_output = expand1darr(vec, 3, 5)


    print(vec_output[0, ])
    print(vec_output[1,])
    print(vec_output[2,])