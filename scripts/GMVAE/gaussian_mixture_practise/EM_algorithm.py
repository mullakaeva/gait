import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs


def initialize_gaussian_mixture(dims, k):
    np.random.seed(10)
    means = np.random.uniform(-1, 1, size=(k, dims))
    pis = np.ones(k) / k  # (k, )
    covs = np.random.uniform(0.5, 2, size=(k, dims, dims))
    for k_idx in range(k):
        covs[k_idx,] = np.eye(dims)
    return means, covs, pis


class GMbyEMalgorithm:
    def __init__(self, num_gaus=2):
        # Initialization
        self.data, self.num_samples, self.dims, self.data_max, self.data_min = self._import_data()
        self.model_means, self.model_covs, self.model_pis = self._initialize_models(num_gaus)
        self.k = num_gaus
        self.current_iter = 0
        self.x1x1, self.x2x2 = self._initialize_space_mesh()

        # Import magnitudes
        self.posteriors = np.zeros((self.num_samples, self.k))
        self.current_likelihood = 0

    def _initialize_models(self, k):
        return initialize_gaussian_mixture(self.dims, k)

    def _import_data(self):
        centers_coors = np.array([
            [0, 0],
            [0, 3],
            [3, 0],
            [0, -3],
            [-3, 0]

        ])
        blobs = make_blobs(
            n_samples=500,
            n_features=2,
            centers=centers_coors,
            cluster_std=[0.5, 0.5, 0.4, 0.5, 0.3]
        )

        loaded_data = blobs[0]

        # Data Normalization
        data = (loaded_data - np.mean(loaded_data, axis=0, keepdims=True)) / np.std(loaded_data, axis=0,
                                                                                    keepdims=True)
        num_samples, dims = data.shape
        data_max, data_min = np.max(data, axis=0), np.min(data, axis=0)
        return data, num_samples, dims, data_max, data_min

    def visualiza_model(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        # Current iteration
        fig.suptitle("%d: Data log likelihood = %f\n Mixture = %s" % (self.current_iter,
                                                                      self.current_likelihood,
                                                                      str(self.model_pis)))
        # Visualize model (gaussian heatmap)
        x_space = np.stack((self.x1x1.flatten(), self.x2x2.flatten())).T  # (samples, 2)
        x_probs = self._evaluate_mixture_gaussian(x_space)  # (samples, k)
        ax.pcolormesh(self.x1x1, self.x2x2, np.sum(x_probs, axis=1).reshape(self.x1x1.shape), cmap="Reds")

        # Visualize data (X)
        ax.scatter(self.data[:, 0], self.data[:, 1], c="k", alpha=0.5, marker="x")

        # Save image
        plt.savefig("model_%d.jpg" % self.current_iter)

    def _initialize_space_mesh(self):
        space_resolution = 200
        x1_space = np.linspace(self.data_min[0], self.data_max[0], space_resolution)
        x2_space = np.linspace(self.data_min[1], self.data_max[1], space_resolution)
        x1x1, x2x2 = np.meshgrid(x1_space, x2_space)
        return x1x1, x2x2

    def _evaluate_mixture_gaussian(self, x):
        # x ~ (samples, dims)
        num_sample = x.shape[0]
        prob = np.zeros((num_sample, self.k))  # (samples, )

        for k_idx in range(self.k):
            prob[:, k_idx] = self.model_pis[k_idx] * multivariate_normal.pdf(x, self.model_means[k_idx],
                                                                             self.model_covs[k_idx])
        return prob  # (samples, k)

    def evaluate_posteriors(self):
        nominator_sum = np.zeros(self.num_samples)
        self.posteriors = np.zeros((self.num_samples, self.k))
        for k_idx in range(self.k):
            nominator = self.model_pis[k_idx] * multivariate_normal.pdf(self.data, self.model_means[k_idx],
                                                                        self.model_covs[k_idx])
            self.posteriors[:, k_idx] = nominator
            nominator_sum += nominator
        self.posteriors = self.posteriors / nominator_sum.reshape(-1, 1)
        return self.posteriors  # (num_samples, k)

    def evaluate_likelilhood(self):
        joint_likelihood = np.zeros((self.num_samples, self.k))
        for k_idx in range(self.k):
            joint_likelihood[:, k_idx] = self.model_pis[k_idx] * multivariate_normal.pdf(self.data,
                                                                                         self.model_means[k_idx],
                                                                                         self.model_covs[k_idx])
        complete_evidence = self.posteriors * joint_likelihood
        self.current_likelihood = np.sum(complete_evidence)

    def optimize_models(self):
        nk = np.sum(self.posteriors, axis=0)  # (k, )

        for k_each in range(self.k):
            posteriors_eachk = self.posteriors[:, k_each].reshape(-1, 1)
            self.model_means[k_each, :] = np.sum(posteriors_eachk * self.data, axis=0) / \
                                          nk[k_each]
            x_diff = self.data - self.model_means[k_each, :].reshape(1, -1)
            self.model_covs[k_each,] = np.dot((posteriors_eachk * x_diff).T, x_diff) / nk[k_each]
            self.model_pis[k_each] = nk[k_each] / self.num_samples

    def optimize_step(self):
        print("=================== Iteration: %d =======================" % self.current_iter)
        for k_idx in range(self.k):
            print("Gaussian %d:" % k_idx)
            print("Mean = \n", self.model_means[k_idx,])
            print("Cov = \n", self.model_covs[k_idx,])
            print("Mix = \n", self.model_pis[k_idx,])
        self.visualiza_model()
        self.evaluate_posteriors()
        self.evaluate_likelilhood()
        self.optimize_models()
        self.current_iter += 1


if __name__ == "__main__":
    optim = GMbyEMalgorithm(num_gaus=5)
    for i in range(40):
        optim.optimize_step()
    optim.visualiza_model()
