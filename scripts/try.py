from common.generator import GaitGeneratorFromDFforTemporalVAE
from common.utils import expand1darr
import numpy as np
if __name__ == "__main__":
    # df_path = "/mnt/data/feas_tasks_phenos_nanMasks_idpatient.pickle"
    # data_gen = GaitGeneratorFromDFforTemporalVAE(df_pickle_path=df_path, m=32, n=128, seed=60)
    #
    # for train_info, test_info, towards_info in data_gen.iterator():
    #     (x_train, x_train_masks, task_train, task_train_masks, pheno_train, pheno_train_masks) = train_info
    #     (x_test, x_test_masks, task_test, task_test_masks, pheno_test, pheno_test_masks) = test_info
    #     (towards_train, towards_test) = towards_info
    #
    #     import pdb
    #     pdb.set_trace()

    foo = np.arange(5)

    foo2 = expand1darr(foo, 5)
    print(foo2.shape)