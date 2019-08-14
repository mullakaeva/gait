from common.generator import GaitGeneratorFromDFforTemporalVAE


df_path = "/mnt/data/full_feas_tasks_phenos_nanMasks_idpatient_leg.pickle"
data_gen = GaitGeneratorFromDFforTemporalVAE(df_pickle_path=df_path,
                                             m=32,
                                             seed=60,
                                             gait_print=True)

for train_data, test_data in data_gen.iterator():
    pass