import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



# Based on https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#load-a-model and 
# https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html

def net():
    #pass
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False # Freeze the convolutional layers of the pre-trained model

    num_ftrs = model.fc.in_features # Get the number of features in the model output 
    model.fc = nn.Linear(num_ftrs, 5) #add a connected layer with 5 classes as output
    
    return model


def model_fn(model_dir):
    logger.info("In model_fn")

    model = net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use GPU to train
    model=model.to(device)

    
    model_file = "model.pth"
    
    with open(os.path.join(model_dir, model_file ), "rb") as f:
        logger.info("LOADING MODEL")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        logger.info('MODEL LOADED')
    model.eval()
    return model




def input_fn(request_body, request_content_type):
    
    logger.info('DESERIALIZING INPUT')

    logger.debug(f'CONTENT-TYPE is: {request_content_type}')
    logger.debug(f'BODY TYPE is: {type(request_body)}')
    
    if request_content_type == 'image/jpeg':
        logger.info('Input file is JPEG')
        return Image.open(io.BytesIO(request_body))
    else:
        # Raise an Exception that the content type is not supported.
        raise Exception('Requested unsupported ContentType in request_content_type: {}'.format(request_content_type))

    
    

# inference
def predict_fn(input_object, model):
    
    logger.info('PREDICTING')
    
    # transform the serialized image to tensor
    testing_transform = transforms.Compose([
        transforms.Resize(256), # Resize and crop images
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    logger.info("TRANSFORMING INPUT")
    input_object = testing_transform(input_object)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_object = input_object.to(device) 
    
    with torch.no_grad():
        logger.info("CALLING MODEL")
        prediction = model(input_object.unsqueeze(0))
    return prediction