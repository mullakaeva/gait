import numpy as np
import matplotlib.pyplot as plt


def generate_sin_data(m_train, m_test, seed=20):
    # Define the whole space
    m_total = m_train + m_test
    np.random.seed(seed)
    noise = np.random.normal(0, 0.2, size=m_total)
    x_whole = np.linspace(0, 2 * np.pi, m_total)
    y_whole = np.sin(x_whole) + noise

    # Random Sampling
    np.random.seed(seed)
    ran_vec = np.random.permutation(m_total)

    # Spliting of train/test set
    x_shuffled, y_shuffled = x_whole[ran_vec], y_whole[ran_vec]
    x_train, y_train = x_shuffled[:m_train], y_shuffled[:m_train]
    x_test, y_test = x_shuffled[m_train:(m_train + m_test)], y_shuffled[m_train:(m_train + m_test)]
    return (x_train, y_train), (x_test, y_test)


def plot_sin_data(m_train, m_test):
    (x_train, y_train), (x_test, y_test) = generate_sin_data(m_train, m_test)
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, c="b", label="train")
    ax.scatter(x_test, y_test, c="r", label="test")
    fig.savefig("sin_data_example.png")
    return None

if __name__=="__main__":
    plot_sin_data(40, 10)
