import time
import torch
from tqdm import tqdm

from config.command_arguments import *
from modules.model import ModelModules
from modules.logger import LogModules
from modules.optimization import OptimModules
from modules.dataloaders import DataloadersModules
from utils.common import init_lr_file, set_lr, save_model


def validate_epoch(device, model, validation_dataloader):
    """
    Calculate validation accuracy after training for one epoch
    
    Returns:
        validation_accuracy (float): The calculated val accuracy

    Side effects:
        prints the accuracy, and the time it took to calculate it.
    """
    print('Validation...')
    pass


def train_epoch(device, writer, model, epoch, optimizer, criterion,
    training_dataloader):
    """
    Trains the model for one epoch

    Returns:
        running_loss: The average loss accross batches
    """
    print(f'Training...')
    model.train()

    running_loss = 0.0
    len_train_loader = len(training_dataloader)
    
    start_time = time.time()
    for i, (data, target) in enumerate(tqdm(training_dataloader)): 

        # Move data to device
        data, target = data.to(device), target.to(device) 
        
        # Forward pass
        outputs = model(data)

        # Calculate loss
        loss = criterion(outputs, target)
        running_loss += loss.detach().item()

        # Backward pass
        loss.backward()

        # Update the weights & biases
        if i % DataModules.Update_step == 0 or i == len_train_loader:

            # Take a step
            optimizer.step()

            # Reset gradient to zero
            optimizer.zero_grad()
        
        # Log progress
        current_batch = epoch * len_train_loader + i
        if current_batch % LogModules.Log_step == 0:
            writer.add_scalar("training loss", loss.item(), current_batch)
    
    # Print progress
    running_loss /= len_train_loader
    run_time = (time.time() - start_time)//60
    print('Training Loss: ', running_loss, ' Time: ', run_time, 'minutes')
    
    # Free gpu ram
    if device != 'cpu':
        torch.cuda.empty_cache()
   
    return running_loss


def run_training(device, writer, model, start_epoch, epochs, optimizer,
    criterion, training_dataloader, validation_dataloader):
    """
    Trains a model for number of epochs
    """
    for i in range(start_epoch, epochs):
        print(f'Epoch Number: {i}')

        training_loss = train_epoch(
            device, writer, model, i, optimizer, criterion, training_dataloader,
        )
        
        validation_accuracy = validate_epoch(
            device, model, validation_dataloader,
        )

        # save chackpoint
        save_model(i, model, optimizer, validation_accuracy)

        # re-set Learning Rates
        set_lr(optimizer, LogConfig.Log_dir)

        # Log progress
        writer.add_scalar("validation accuracy", validation_accuracy)


if __name__ == "__main__":
    # define the used device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    # import the training components
    start_epoch = 0
    writer = LogModules.Writer
    model = ModelModules.Model
    epochs = DataloadersModules.Num_epochs
    optimizer = OptimModules.Optimizer
    criterion = OptimModules.Criterion
    training_dataloader = DataloadersModules.Training_Dataloader
    validation_dataloader = DataloadersModules.Validation_Dataloader

    # move componenets to the used device
    model.to(device)
    criterion.to(device)

    # load checkpoint if available
    checkpoint = LogModules.Checkpoint
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # create the learning rate file
    init_lr_file(optimizer.param_groups[-1]["lr"], LogConfig.Log_dir)

    # start training
    run_training(
        device,
        writer,
        model,
        start_epoch,
        epochs,
        optimizer,
        criterion,
        training_dataloader,
        validation_dataloader,
    )