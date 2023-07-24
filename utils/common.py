import os
import json
import torch


def save_model(epoch, model, optimizer, accuracy, log_dir):
    """
    Saves trained DL Models
    
    Args:
        model (pytorch DL model): DL model
        epoch (int): number of current epoches
        final (bool, optional): indicate if final. Defaults to False.
    """
    # create output directory if doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    save_path = os.path.join(
        log_dir,
        f"checpoint-E({epoch})-A({accuracy}).pth"
    )

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)


def init_lr_file(lr, log_dir):
    """
    Create the file that stores the the learning rate.

    Args:
        lr (_type_): _description_
        log_dir (_type_): _description_
    """
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'LR.json'), "w+") as outfile:
        outfile.write(json.dumps({'lr': lr}))


def set_lr(optimizer, log_dir):
    """
    Reset the learning rate of the optimizer based on the value in LR.json.
    
    Args:
        optimizer: the training optimizer
        log_dir: the directory that contains the file that stores the LR
    """
    with open(os.path.join(log_dir, 'LR.json'), 'r') as openfile:
        # Read value of LR
        json_object = json.load(openfile)
        lr = json_object['lr']
        # Set the LR
        for g in optimizer.param_groups:
            g['lr'] = lr
    print(f'Learning Rate: {lr}')
