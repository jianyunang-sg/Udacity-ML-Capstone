#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import json
import logging
import os
import sys

# Load truncated image files
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Import SMDebug framework class
import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #pass
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    hook.set_mode(smd.modes.EVAL) # set the hook for testing phase
    
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        # move to GPU
        inputs=inputs.to(device)
        labels=labels.to(device)
        
        outputs=model(inputs) # forward pass: compute predicted outputs by passing inputs to the model
        loss=criterion(outputs, labels) # calculate the loss

        _, preds = torch.max(outputs, 1) # get the predictions
        running_loss += loss.item() * inputs.size(0) # update total loss per epoch
        running_corrects += torch.sum(preds == labels.data).item() # update correct predictions per epoch

    total_loss = running_loss / len(test_loader.dataset) # calculate loss for test dataset
    total_acc = running_corrects/ len(test_loader.dataset)  # calculate accuracy for test dataset
    #print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * running_corrects / len(test_loader.dataset)
        )
    )    
    
def train(model, train_loader, validation_loader, criterion, optimizer, epoch, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #pass
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    hook.set_mode(smd.modes.TRAIN) # Set the hook for the training phase
    
        
    for phase in ['train', 'valid']:
        print(f"Epoch {epoch}, Phase {phase}")
        if phase=='train':
            model.train()
                
        else:
            model.eval()
                
        running_loss = 0.0
        running_corrects = 0
        running_samples=0

        for step, (inputs, labels) in enumerate(image_dataset[phase]):
            # move to GPU
            inputs=inputs.to(device)
            labels=labels.to(device)
                
            outputs = model(inputs) # forward pass: compute predicted outputs by passing inputs to the model      
            loss = criterion(outputs, labels) # calculate the batch loss
                
            if phase=='train':
                optimizer.zero_grad() # clear the gradients of all optimized variables
                loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step() # perform a single optimization step (parameter update)

            _, preds = torch.max(outputs, 1) # get the predictions
            running_loss += loss.item() * inputs.size(0) # update total loss per epoch
            running_corrects += torch.sum(preds == labels.data).item() # update correct predictions per epoch
            running_samples+=len(inputs) # number of data inputs processed
            if step % 50  == 0: # print out the loss and accuracy
                accuracy = running_corrects/running_samples
                print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        step * len (inputs),
                        len(image_dataset[phase].dataset),
                        100.0 * (running_samples / len(image_dataset[phase].dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*accuracy,
                    )
                )

        epoch_loss = running_loss / running_samples # fraction of loss for epoch
        epoch_acc = running_corrects / running_samples # accuracy for epoch
            
        if phase=='valid': # check if epoch loss is decreasing, else exit training
            if epoch_loss<best_loss:
                best_loss=epoch_loss
            else:
                loss_counter+=1

    #if loss_counter==1: # exit training if performance does not improve
    #    break
    
    return model

    
    
def net():
    #pass
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False # Freeze the convolutional layers of the pre-trained model

    num_ftrs = model.fc.in_features # Get the number of features in the model output 
    model.fc = nn.Linear(num_ftrs, 5) #add a connected layer with 5 classes as output
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net()
    
    # Register the SMdebug hook to save output tensors
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use GPU to train
    model=model.to(device)
    print(f"Running on Device {device}")
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
   
    # Set up the transforms and  data loaders
    training_transform = transforms.Compose([
        transforms.Resize(256), # Resize and crop images
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(256), # Resize and crop images
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testing_transform = transforms.Compose([
        transforms.Resize(256), # Resize and crop images
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read the S3 image directory 
    train_dir = os.environ['SM_CHANNEL_TRAIN']
    val_dir = os.environ['SM_CHANNEL_VAL']
    test_dir = os.environ['SM_CHANNEL_TEST']

    # Prepare and load the datasets
    train_dataset = torchvision.datasets.ImageFolder(root = train_dir, transform=training_transform)
    val_dataset = torchvision.datasets.ImageFolder(root = val_dir, transform=validation_transform)
    test_dataset = torchvision.datasets.ImageFolder(root = test_dir, transform=testing_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle = False)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''    
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, validation_loader, criterion, optimizer, epoch, device, hook)    
        test(model, test_loader, criterion, device, hook)
    '''
    TODO: Test the model to see its accuracy
    '''    
    
       
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
     # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir-train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--data-dir-test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--data-dir-val", type=str, default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
