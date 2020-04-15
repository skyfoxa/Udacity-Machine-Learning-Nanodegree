import utils
import json
from PIL import Image
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model   
    pil_image = Image.open(image)
    w, h = pil_image.size
    
    # SCALE
    min_size = 256
    ratio = w / h
    if w < h and w < min_size :
        w = min_size
        h = w / ratio
    elif w > h and h < min_size:
        h = min_size
        w = h * ratio
    pil_image = pil_image.resize((w, h))
    
    # CROP
    center = (w/2, h/2)
    cropSize_half = 224 / 2
    pil_image = pil_image.crop((center[0] - cropSize_half, center[1] - cropSize_half, center[0] + cropSize_half, center[1] + cropSize_half))
        
    np_image = np.array(pil_image)
    np_image = np_image / np.array([255])


    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np.transpose(np_image, (2, 0, 1))

    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predictWith(image_path, model, topk=5, isGPU=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if isGPU else "cpu")
    
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        model.eval()
        model.to(device)
        image = process_image(image_path)
        image = torch.from_numpy(image).unsqueeze(0).float()
        image = image.to(device)     
        outs = model(image)
        
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}

        top_p, top_class = nn.functional.softmax(outs).topk(topk, dim=1)
        top_p = sum(top_p.tolist(), [])
        top_class_idx = list(map(lambda c: idx_to_class[c],sum(top_class.tolist(), [])))
    
        return top_p, top_class_idx
    
def predict(categoryFilePath, checkpointPath, img_path, topK=5, isGPU=False):
    with open(categoryFilePath, 'r') as f:
        cat_to_name = json.load(f)
        
    model = utils.loadModel(checkpointPath)

    probs, classes = predictWith(img_path, model, topK, isGPU)

    classes = list(map(lambda c: cat_to_name[c],classes))
    
    return (probs, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flower category')
    parser.add_argument('input', action="store", type=str)
    parser.add_argument('checkpoint', action="store", type=str)
    parser.add_argument('--top_k', type=int, default=5, help='Number of top probable categories of the prediction')
    parser.add_argument('--category_names', type=str, default="cat_to_name.json", help='JSON mapping category to name')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    

    args = parser.parse_args()
    print(args)
    (probs, classes) = predict(args.category_names, args.checkpoint, args.input, args.top_k, args.gpu)
    
    print(f"Prob: {list(map(lambda p: '%.2f' % (p*100.0),probs))}")
    print(f"Class: {classes}")
    

        
     