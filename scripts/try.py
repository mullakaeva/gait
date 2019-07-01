import numpy as np
import torch


if __name__ == "__main__":
    num_samples = 10
    gradient_mulitplier = np.random.randint(5, 10, (num_samples,))
    mask = np.random.randint(0, 2, (num_samples,))

    loss = torch.randn(num_samples, requires_grad=True)
    gradient_mulitplier = torch.from_numpy(gradient_mulitplier).float()
    mask = torch.from_numpy(mask).float()

    # multiplication
    final_loss = torch.mean(loss * gradient_mulitplier * mask)
    final_loss.backward()

    # Look at gradient
    print("mask:\n", mask)
    print("Gradient multiplier:\n", gradient_mulitplier)
    print("loss's grad\n", loss.grad)







