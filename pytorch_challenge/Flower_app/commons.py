import io
from torchvision import models
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from collections import OrderedDict

def get_model():
	checkpoint_path='classifier.pth'
	model = models.densenet161(pretrained=True)
	#model.classifier= nn.Linear(2208, 102)
	model.load_state_dict(torch.load(
		checkpoint_path, map_location='cpu', strict=False))
	model.eval()
	return model

def get_tensor(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(226),
		transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	image= Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)

def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    
    if chpt['model'] == 'densenet161':
        model = models.densenet161(pretrained=False)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture note recognized")
      
    
    model.class_to_idx = chpt['class_to_idx']
    from collections import OrderedDict
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2208, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.4)),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'], strict=False)
    model.eval()
    return model