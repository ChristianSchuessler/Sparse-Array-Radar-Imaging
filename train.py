
import os
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from .FullRadarCubeDataset import FullRadarCubeDataset, FullRadarCubeDatasetConfig
from .RadarUnet import RadarUNet
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from sigproc import *
from torch.optim.lr_scheduler import MultiStepLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

training_artifcats_dir = "/"

tx_positions = np.asarray([np.array([-5.0e-3, 0.0, 0]), 
                np.array([-3.0e-3, 0.0, 0]),
                np.array([-1.0e-3, 0.0, 0])])

rx_positions = np.asarray([np.array([0.0e-3, 0.0, 0.0]),
            np.array([6.0e-3, 0.0, 0.0]),
            np.array([12.0e-3, 0.0, 0.0]),
            np.array([18.0e-3, 0.0, 0.0]),
            np.array([24.0e-3, 0.0, 0.0]),
            np.array([30.0e-3, 0.0, 0.0]),
            np.array([36.0e-3, 0.0, 0.0]),
            np.array([42.0e-3, 0.0, 0.0]),
            np.array([48.0e-3, 0.0, 0.0]),
            np.array([54.0e-3, 0.0, 0.0]),
            np.array([60.0e-3, 0.0, 0.0]),
            np.array([66.0e-3, 0.0, 0.0]),
            np.array([72.0e-3, 0.0, 0.0]),
            np.array([78.0e-3, 0.0, 0.0]),
            np.array([84.0e-3, 0.0, 0.0]),
            np.array([90.0e-3, 0.0, 0.0])])


rx_dict = {"sparse16":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "sparse4":[0,1,10,15],
        "sparse3":[0,5,10],
        "sparse6":[0, 1, 6, 10, 13, 15]}

virt_indices_dict = {}
steering_vectors_dict = {}
tx_indices = [0,1,2]
for key in rx_dict.keys():
    # construct steering vector
    steering_vector = create_steering_vector(tx_positions[:,0], rx_positions[rx_dict[key]][:,0])
    steering_vectors_dict[key] = steering_vector
    
    # construct indices for virtual array (array occupation)
    rx_indices = np.array(rx_dict[key])*3
    virt_indices = []
    for rx_index in rx_indices:
        for tx_index in tx_indices:
            virt_indices.append(rx_index+tx_index)
    virt_indices_dict[key] = virt_indices

selected_antenna_config = "sparse16"

def input_transform_channel(file, idx)-> torch.tensor:
    """
    This method transforms the input data of the neural network, by 
    requiring a h5-file handle and the current data index
    """

    # range-channel-data (range-Doppler-fft + Doppler selection applied, but channel data unchanged)
    rc_image_np = np.array(file.get(f'rc_data_{idx:06d}'), dtype=np.complex64)

    rc_image_np = rc_image_np[virt_indices_dict[selected_antenna_config], :]
    rc_image = rc_image_np.T[:, np.newaxis, :]
    rc_image = torch.tensor(rc_image, device=device)

    steering_vector = torch.tensor(steering_vectors_dict[selected_antenna_config], device=device, dtype=torch.complex64)
    steering_vector = torch.unsqueeze(steering_vector.T, 0)
    ra_image = rc_image@torch.conj(steering_vector)
    ra_image = torch.squeeze(ra_image, 1).T
    ra_image_abs = torch.log10(torch.abs(ra_image))
    ra_image_abs = normalize_data(ra_image_abs)

    rc_image_T = torch.swapaxes(rc_image, 1, 2)
    cov_mat_image_torch = rc_image_T@torch.conj(rc_image)
    cov_mat_image_torch /= torch.linalg.norm(cov_mat_image_torch)

    max_cov_size = min(29, rc_image_np.shape[0])
    cov_mat_image_torch = cov_mat_image_torch[:, :max_cov_size, :max_cov_size]
    idx = torch.triu_indices(*cov_mat_image_torch.shape[1:])
    cov_mat_line = cov_mat_image_torch[:, idx[0], idx[1]].T

    cov_mat_image = torch.zeros(ra_image.shape, dtype=torch.complex64, device=device)
    cov_mat_image[:cov_mat_line.shape[0], :] = cov_mat_line

    input_tensor = torch.stack((ra_image_abs, torch.angle(ra_image), torch.real(cov_mat_image), torch.imag(cov_mat_image), torch.angle(cov_mat_image)))
    return input_tensor


def target_transform_simple(file, idx) -> torch.tensor:
    """
    This method transforms the target/output data of the neural network, by 
    requiring a h5-file handle and the current data index
    """

    # expect image, which was created by a matched filter in range-sin(angle) coordinates
    ra_image = np.array(file.get(f'ra_data_matched_{idx:06d}'), dtype=np.float32)
        
    ra_image -= np.min(ra_image)
    ra_image /= (np.max(ra_image)+1e-2)
    scale_factor = 10
    ra_image = ra_image*scale_factor

    return torch.tensor(ra_image)

def train(experiment_name, model, num_epochs, learning_rate, out_directory):

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    
    # Tensorboard writer
    writer = SummaryWriter(f'runs/{experiment_name}')
    model_param_name = f"{experiment_name}.params"
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    avg_test_loss_before = 10
    batch_step = 0
    epoch_step = 0
    avg_train_loss = 10

    scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}] -> last-test-loss: {avg_test_loss_before}, last-train-loss: {avg_train_loss}, learning_rate: {scheduler.get_last_lr()}")

        train_data_set.shuffle_data(epoch)
        model.train()
        train_loss = 0
        for batch_idx, (X, y) in enumerate(train_data_loader):

            X = X.to(device)
            y = y.to(device)
            # Forward pass
            pred = model(X)
            if batch_idx == 0: 
                fig, axes = plt.subplots(max(2,train_data_loader.batch_size), 3, figsize=(15, 4*train_data_loader.batch_size))
                for image_idx in range(train_data_loader.batch_size):

                    plt1 = axes[image_idx,0].imshow(X[image_idx,0].cpu().detach().numpy())
                    plt.colorbar(plt1 ,ax=axes[image_idx,0])

                for image_idx in range(train_data_loader.batch_size):
                    plt1 = axes[image_idx,1].imshow(pred[image_idx,0].cpu().detach().numpy())
                    plt2 = axes[image_idx,2].imshow(y[image_idx].cpu().detach().numpy())
                    plt.colorbar(plt1 ,ax=axes[image_idx,1])
                    plt.colorbar(plt2 ,ax=axes[image_idx,2])
                writer.add_figure("Predicted-Train", fig, global_step=epoch_step)
                plt.close()

            N = 4 # increase batch size even with insufficient video memory by accumulating gradients
            loss = loss_fn(torch.squeeze(pred, dim=1), y) / N
            
            loss.backward()
            if (batch_idx+1) % N == 0:
                train_loss += loss.item()
                optimizer.step()
                writer.add_scalar("Training loss - Batch", loss, global_step=batch_step)
                batch_step += 1

                max_value = 0
                total_norm = 0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    param_norm = p.grad.data.norm(2)
                    if param_norm > max_value:
                        max_value = p.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                if batch_idx % 100 == 0:
                    print(f"batch-idx: {batch_idx:06d} -> loss: {loss.item():.4f} -> grad_sum: {total_norm:.4f} -> grad_max: {max_value:.4f}")

                optimizer.zero_grad()

            if batch_idx % 1000 == 0:
                model_filename = os.path.join(out_directory, f"epoch_{epoch}_{batch_idx}_{model_param_name}")
                torch.save(model.state_dict(), model_filename)
                print(f"saved model {model_filename}")

        # add mean avg loss for current epoch
        avg_train_loss = train_loss / len(train_data_loader)

        # Evaluate the model at the beginning of each epoch
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(test_data_loader):

                X = X.to(device)
                y = y.to(device)
                pred = model(X)

                if batch_idx == 0:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    plt0 = axes[0].imshow(X[0,0].cpu().detach().numpy())
                    plt.colorbar(plt0 ,ax=axes[0])

                    plt1 = axes[1].imshow(pred[0,0].cpu().detach().numpy())
                    plt2 = axes[2].imshow(y[0].cpu().detach().numpy())
                    plt.colorbar(plt1 ,ax=axes[1])
                    plt.colorbar(plt2 ,ax=axes[2])
                    writer.add_figure("Predicted-Test", fig, global_step=epoch_step)
                    plt.close()
                test_loss += loss_fn(torch.squeeze(pred, dim=1), y).item()
        
       
        avg_test_loss = test_loss / len(test_data_loader)
        torch.save(model.state_dict(), os.path.join(out_directory, f"epoch_{epoch}_{model_param_name}"))
        avg_test_loss_before = avg_test_loss

        # plot both in tensorboard
        loss_dict = {"avg_train_loss":avg_train_loss, "avg_test_loss":avg_test_loss}
        writer.add_scalars("Loss-Epoch", loss_dict, global_step=epoch_step)

        epoch_step += 1
        scheduler.step()

antenna_configs = ["sparse16"]
for antenna_config_idx, antenna_config in enumerate(antenna_configs):

    selected_antenna_config = antenna_config
    experiment_name = f"U-Net_{antenna_config}_pub"

    print(f"run expriment: {experiment_name}")
    selected_rx = rx_dict[antenna_config]
    train_data_config = FullRadarCubeDatasetConfig()

    train_data_config.data_set_size = 18500
    train_data_config.number_valid_samples = 500
    train_data_config.number_test_samples = 500
    train_data_config.number_train_samples = 17500
    train_data_config.input_filename = f"input_data.h5"
    train_data_config.target_filename =  f"target_data.h5"
    
    train_data_config.mode = "train"
    train_data_config.input_load_callback = input_transform_channel
    train_data_config.target_load_callback = target_transform_simple

    train_data_set = FullRadarCubeDataset(train_data_config)
    eval_data_config = FullRadarCubeDatasetConfig()
    eval_data_config.data_set_size = 18500
    eval_data_config.number_valid_samples = 500
    eval_data_config.number_test_samples = 500
    eval_data_config.number_train_samples = 17500
    eval_data_config.input_filename = f"input_data.h5"
    eval_data_config.target_filename = f"target_data.h5"
    eval_data_config.mode = "valid"

    eval_data_config.input_load_callback = input_transform_channel
    eval_data_config.target_load_callback = target_transform_simple

    eval_data_set = FullRadarCubeDataset(eval_data_config)
    x, y = train_data_set[0]

    # initialize model and training parameters
    train_data_loader = DataLoader(train_data_set, batch_size=2)
    test_data_loader = DataLoader(eval_data_set, batch_size=1)

    torch.random.manual_seed(99)
    model = RadarUNet(y.shape, x.shape[0]).to(device)

    learning_rate =1e-4
    num_epochs = 6
        
    train(experiment_name, model, num_epochs, learning_rate)




