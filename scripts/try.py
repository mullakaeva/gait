import pandas as pd
import numpy as np
import os
from glob import glob
from common.utils import task2idx, pheno2idx, load_df_pickle

if __name__ == "__main__":
    # df_save_path = "/mnt/data/labels/fn_tasks_phenos.pkl"
    # df = load_df_pickle(df_save_path)
    #
    # print(df.columns)
    # print(df.fn_mp4.head())
    # print(df.idpatient.iloc[0])
    # print(type(df.idpatient.iloc[0]))

    df_load_path = "/mnt/data/feas_tasks_phenos_nanMasks_idpatient.pickle"

    df = load_df_pickle(df_load_path)

    ids = np.unique( df[ np.isnan(df["idpatients"]) != True]["idpatients"])
    print(ids)


    # print('Number of unique videos synched to subject (primary diagnosis only): %d.' %
    #       np.unique(df.fn_mp4).shape[0])
    #
    # unqdiag, unqcounts = np.unique([str(x) for x in df.diagnosis_label[
    #     df.diagnosis_status == 'validated']], return_counts=True);
    # unqdiags = pd.DataFrame();
    # unqdiags['diag'] = unqdiag.tolist()
    # unqdiags['counts'] = unqcounts.tolist()
    #
    # unqphen, unqcounts = np.unique([str(x) for x in df.phenotyp_label], return_counts=True);
    # unqphens = pd.DataFrame();
    # unqphens['phen'] = unqphen.tolist()
    # unqphens['counts'] = unqcounts.tolist()
    #
    # print(unqphens)

