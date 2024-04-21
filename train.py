"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from model import Model
from argparse import ArgumentParser
from loss_functions import DiceLoss, CombinedLoss
import numpy as np
import torch
import torch.optim as optim
import torchvision
import wandb
import time
from sklearn.metrics import f1_score
from transforms import CityscapesDataset


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    wandb.init(
        project = "5LSM0 project, hyperparameter training",
    )

    # Hyperparameters
    num_workers = 12
    batch_size = 16
    num_epochs = 25
    learning_rate = 0.001
    patience = 5
    smooth = 1e-8
    print(f"Learning rate={learning_rate}")

    # Load dataset
    original_dataset = torchvision.datasets.Cityscapes(
        root=args.data_path,
        split = 'train',
        mode='fine',
        target_type='semantic',
    )

    # Split dataset into train, validation and test set
    dataset_size = len(original_dataset)
    train_size = int(0.8 * dataset_size)
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_set = torch.utils.data.Subset(original_dataset, train_indices)
    val_set = torch.utils.data.Subset(original_dataset, val_indices)

    trainset = CityscapesDataset(train_set, augmentation=True)
    valset = CityscapesDataset(val_set, augmentation=True)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                                              num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=True,
                                              num_workers=num_workers)

    # Define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    # Define loss function, optimizer and scheduler
    loss_function = CombinedLoss(smooth=smooth)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize parameters for loop
    best_val = float('inf')
    patience_counter = 0

    # Training/validation loop
    for epoch in range(num_epochs):
        # The training loop
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, targets in trainloader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            outputs = model(inputs)
            targets = targets.long().squeeze()
            targets = targets.to(device)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()      
            running_loss += loss.item()

        end_time = time.time() - start_time
        train_epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {train_epoch_loss:.4f}, Time: {end_time}")

        torch.save(model.state_dict(), "model.pth")

        # The validation loop
        model.eval()
        running_loss = 0.0
        start_time = time.time()
        f1_scores = []

        with torch.no_grad():
            for inputs, targets in valloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                targets = targets.long().squeeze()
                targets = targets.to(device)
                loss = loss_function(outputs, targets)
                running_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                predictions_flat = predictions.cpu().view(-1).numpy()
                targets_flat = targets.cpu().view(-1).numpy()
                batch_f1 = f1_score(targets_flat, predictions_flat, average='macro')
                f1_scores.append(batch_f1)
                
        end_time = time.time() - start_time
        val_epoch_loss = running_loss / len(valloader)
        average_f1 = np.mean(f1_scores)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation loss: {val_epoch_loss:.4f}, Time: {end_time:1f}")
        print(f'Average F1 score: {average_f1:.4f}')

        wandb.log({"Train loss": train_epoch_loss, "Validation loss": val_epoch_loss, "F1 score": average_f1})

        torch.save(model.state_dict(), "model.pth")
        # Save model with lowest validation loss
        if val_epoch_loss < best_val:
            best_val = val_epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), "model_scaling.pth")
        # Stop after patience epochs
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    wandb.finish()

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
