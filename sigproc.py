import numpy as np
import torch

def create_steering_vector(tx_positions : np.ndarray, rx_positions : np.ndarray, sample_vector = None) -> np.ndarray:
    """
    Create a steering vector, which samples in the sine domain from -0.85 to 0.85 if not sample_vector is given
    """

    if sample_vector is None:
        angular_size=450
        fc = 77e9
        c=3e8
        wavelength = c/fc
        sample_vector = np.linspace(-0.85, 0.85, angular_size, endpoint=False)
        sample_vector = np.flip(sample_vector)[:, np.newaxis]

    if sample_vector.ndim == 1:
        sample_vector = sample_vector[:, np.newaxis]
    
    input_virtual_positions = np.empty((1, len(tx_positions)*len(rx_positions)))
    for pos_idx, rx_pos in enumerate(rx_positions):
        virt_pos = rx_pos + tx_positions
        input_virtual_positions[0, pos_idx*len(tx_positions):(pos_idx+1)*len(tx_positions)] = virt_pos

    steering_vector = np.exp(2.0j*np.pi/wavelength*(sample_vector@input_virtual_positions))
    return steering_vector


def create_normalized_cov_mat(channel_signal : np.ndarray):
    """
    expects a 1D-signal and creates a normalized sample covariance matrix
    """

    channel_signal_ext = channel_signal[:, np.newaxis]
    cov_mat = channel_signal_ext@np.conj(channel_signal_ext.T)
    cov_mat /= np.linalg.norm(cov_mat)

    return cov_mat

def normalize_data(input : np.ndarray):
    """
    normalize data to the range 0->1 for numpy arrays and pytorch tensors
    """

    if type(input) == np.ndarray:
        min_val = np.min(input)
        max_val = np.max(input)
        input_out = (input - min_val) / (max_val - min_val)
        return input_out
    elif type(input) is torch.Tensor:
        min_val = torch.min(input)
        max_val = torch.max(input)
        input_out = (input - min_val) / (max_val - min_val)
        return input_out
    else:
        raise Exception("Data type not supported. Supported is np.ndarray and torch.tensor")

