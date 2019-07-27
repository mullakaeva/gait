from common.utils import expand1darr, load_df_pickle
import skvideo.io as skv

if __name__ == "__main__":
    df_path = "/mnt/JupyterNotebook/interactive_latent_exploration/data/conditional-0.0001-latent_space.pickle"
    df = load_df_pickle(df_path)
    print(df.columns)

