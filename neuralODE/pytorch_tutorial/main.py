from gen_data import generate_sin_data
from example_model import SimpleFCModel
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_iterations = 2000
    learning_rate = 0.001
    num_hid = 100

    (X_train, Y_train), (X_test, Y_test) = generate_sin_data(80, 10)
    X_train, Y_train, X_test, Y_test = X_train.reshape(-1, 1),\
                                       Y_train.reshape(-1, 1),\
                                       X_test.reshape(-1, 1),\
                                       Y_test.reshape(-1, 1)
    model = SimpleFCModel(x_train=X_train,
                          y_train=Y_train,
                          num_hidden_units=num_hid)
    model.train(num_iterations, learning_rate)
    y_test_pred, y_test_loss = model.forward_pass(X_test, Y_test)
    y_pred_first, loss_first = model.get_record(0)
    y_pred_last, loss_last = model.get_record(num_iterations-1)

    fig, ax = plt.subplots(2, 2)

    # Plot first iter's results
    ax[0, 0].scatter(X_train, Y_train, c="b", label="train")
    ax[0, 0].scatter(X_train, y_pred_first, c = "r", label="1st train_result")
    ax[0, 0].set_title("loss = {}".format(loss_first))
    ax[0, 0].legend()

    # Plot last iter's results
    ax[0, 1].scatter(X_train, Y_train, c="b", label="train")
    ax[0, 1].scatter(X_train, y_pred_last, c="r", label="last train_result")
    ax[0, 1].set_title("loss = {}".format(loss_last))
    ax[0, 1].legend()

    # Plot test results
    ax[1, 0].scatter(X_train, Y_train, c="b", label="train")
    ax[1, 0].scatter(X_test, Y_test, c="g", label="test")
    ax[1, 0].scatter(X_test, y_test_pred, c="r", label="test prediction")
    ax[1, 0].set_title("loss = {}".format(y_test_loss))
    ax[1, 0].legend()

    # hide axis
    ax[1, 1].axis("off")

    # Save figure
    fig.savefig("Fitting Results.png")
    # plt.show()