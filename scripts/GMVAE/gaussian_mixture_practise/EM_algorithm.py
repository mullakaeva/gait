import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def oldfaithful_data(data_path):
    df = pd.read_csv(data_path)
    data = np.asarray(df)
    return data

def visualize_oldfaithful(data):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1])
    plt.savefig("OldFaithful_data_vis.png")

def initialize_gaussian_mixture(dims, k):
    means = np.zeros((k, dims))
    pis = np.ones(k) / k  # (k, )
    covs = np.zeros((k, dims, dims))
    for k_idx in range(k):
        covs[k_idx, ] = np.eye(dims)
    return (means, covs, pis)


class GMbyEMalgorithm:
    def __init__(self, data_path, num_gaus=2):
        # Initialization
        self.data, self.dims, self.data_max, self.data_min = self._import_data(data_path)
        self.model_means, self.model_covs, self.model_pis = self._initialize_models(num_gaus)
        self.k = num_gaus
        self.current_iter = 0
        self.x1x1, self.x2x2 = self._initialize_space_mesh()


    def _initialize_models(self, k):
        return initialize_gaussian_mixture(self.dims, k)

    def _import_data(self, data_path):
        loaded_data = oldfaithful_data(data_path)
        # Data Normalization
        data = (loaded_data - np.mean(loaded_data, axis=0, keepdims=True))/np.std(loaded_data, axis=0,
                                                                                       keepdims=True)
        dims = data.shape[1]
        data_max, data_min = np.max(data, axis=0), np.min(data, axis=0)
        return data, dims, data_max, data_min

    def visualiza_model(self):
        fig, ax = plt.subplots(figsize=(12,12))
        # Current iteration
        fig.suptitle("Iteration = %d" % self.current_iter)
        # Visualize model (gaussian heatmap)
        x_space = np.stack((self.x1x1.flatten(), self.x2x2.flatten())).T # (samples, 2)
        x_probs = self._evaluate_mixture_gaussian(x_space) # (samples, 1)
        x_probs = x_probs.reshape(self.x1x1.shape)
        # ax.pcolormesh(self.x1x1, self.x2x2, x_probs, cmap="hot")
        im = ax.imshow(x_probs)
        plt.colorbar(im)

        # Visualize data (X)
        # ax.scatter(self.data[:, 0], self.data[:, 1], c="k")
        # Save image
        plt.savefig("model_%d.png"%self.current_iter)

    def _initialize_space_mesh(self):
        space_resolution = 50
        x1_space = np.linspace(self.data_min[0], self.data_max[0], space_resolution)
        x2_space = np.linspace(self.data_min[1], self.data_max[1], space_resolution)
        x1x1, x2x2 = np.meshgrid(x1_space, x2_space)
        return x1x1, x2x2

    def _evaluate_mixture_gaussian(self, x):
        # x ~ (samples, dims)
        prob = np.zeros(x.shape[0]) # (samples, )
        for k_idx in range(self.k):
            denominator = ((2 * np.pi) ** (self.dims/2) ) * (np.linalg.det(self.model_covs[k_idx, ])**(1/2))
            x_diff = (x - self.model_means[k_idx, ].reshape(1, -1)).reshape(self.dims, -1)
            exponent = -(1/2) * np.matmul(np.matmul(x_diff.T, np.linalg.inv(self.model_covs[k_idx])), x_diff)
            prob_each = self.model_pis[k_idx,] * np.exp(exponent)/denominator
            prob += np.diag(prob_each)
        return prob  # (samples, )












if __name__ == "__main__":
    data_path = "OldFaithful.csv"
    optim = GMbyEMalgorithm(data_path=data_path,
                            num_gaus=1)

    optim.visualiza_model()
    print(optim.data)
    print("Data's shape = ", optim.data.shape)
