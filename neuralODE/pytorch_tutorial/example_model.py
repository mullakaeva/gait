import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, input_dims=1, hidden_dims=50, output_dims=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class SimpleFCModel:
    def __init__(self, x_train, y_train, num_hidden_units):
        """

        Parameters
        ----------
        x_train : Numpy array with shape (num_samples, dimension)
            Input data.
        y_train : Numpy array with shape (num_samples, dimension)
            Target data.
        num_hidden_units: int
            Number of hidden units.
        """
        self.num_samples, self.num_hidden_units = x_train.shape[0], num_hidden_units
        self.input_dims, self.output_dims = x_train.shape[1], y_train.shape[1]
        assert self.num_samples == y_train.shape[0]
        self.x_train = torch.from_numpy(x_train).float()
        self.y_train = torch.from_numpy(y_train).float()
        self.net = Net(self.input_dims, self.num_hidden_units, self.output_dims)
        self.recorder = {
            "y_pred": [],
            "loss": []
        }
        self.criterion = None

    def train(self, num_iterations, learning_rate=0.001):
        optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        self.criterion = nn.MSELoss()
        for i in range(num_iterations):
            y_pred = self.net(self.x_train)
            loss = self.criterion(y_pred, self.y_train)
            self._record_fitting_results(y_pred, loss)
            loss.backward()
            optimizer.step()
            print("Iteration %d/%d | Loss = %0.5f" % (i, num_iterations, loss))

    def _record_fitting_results(self, y_pred, loss):
        self.recorder["y_pred"].append(y_pred)
        self.recorder["loss"].append(loss)

    def get_record(self, idx):
        recorded_y_pred = self.recorder["y_pred"][idx]
        recorded_loss = self.recorder["loss"][idx]
        return recorded_y_pred.detach().numpy(), recorded_loss.detach().numpy()

    def forward_pass(self, x_test, y_test):
        x_tensor = torch.from_numpy(x_test).float()
        y_tensor = torch.from_numpy(y_test).float()
        out = self.net(x_tensor)
        loss = self.criterion(out, y_tensor)

        return out.detach().numpy(), loss.detach().numpy()


if __name__ == "__main__":
    pass
