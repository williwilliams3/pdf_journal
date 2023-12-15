import numpy as np
import matplotlib.pyplot as plt
import os

# Taken from https://github.com/eemlcommunity/PracticalSessions2023/tree/main/diffusion


def create_checkerboard_data(num_datapoints: int, rng_seed: int = 42) -> np.ndarray:
    """Checkerboard dataset."""
    rng = np.random.RandomState(rng_seed)
    x1 = rng.rand(num_datapoints) * 4 - 2
    x2 = (
        rng.rand(num_datapoints)
        - rng.randint(0, 2, [num_datapoints]) * 2.0
        + np.floor(x1) % 2
    )
    data = np.stack([x1, x2]).T * 2
    data = (data - data.mean(axis=0)) / data.std(axis=0)  # normalize
    return data.astype(np.float32)


def create_directory(directory_path):
    """
    Create a directory if it does not exist.

    Parameters:
    - directory_path (str): The path of the directory to create.
    """
    # Check if the directory exists
    if not os.path.exists(directory_path):
        try:
            # Create the directory
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' already exists.")


def plot_data(data: np.ndarray):
    plt.figure(figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    train_data = create_checkerboard_data(1000000)
    create_directory("data")
    np.save("data/checkboard.npy", train_data)

    # Sanity check
    plot_data(train_data[:5000])
