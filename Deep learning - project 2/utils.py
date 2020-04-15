import torch
from torch import nn
from torchvision import transforms, datasets, models
from collections import OrderedDict

def loadData(trainDir, validDir, batchSize=64):
    
    data_transforms = OrderedDict([
        ("train", transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])),
        ("valid", transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))
        ])

    image_datasets = OrderedDict([
        ("train", datasets.ImageFolder(trainDir, data_transforms["train"])),
        ("valid", datasets.ImageFolder(validDir, data_transforms["valid"])),
    ])


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = OrderedDict([
        ("train", torch.utils.data.DataLoader(image_datasets["train"], batch_size=batchSize, shuffle=True)),
        ("valid", torch.utils.data.DataLoader(image_datasets["valid"], batch_size=batchSize)),
    ])
    
    return dataloaders, image_datasets['train'].class_to_idx



def buildModel(state=None, hidden=512, architecture="vgg16"):
    arch = getattr(models, architecture)
    model = arch(pretrained=True)

    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(4096, hidden),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(hidden, 102),
        nn.LogSoftmax(dim=1)
    )

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier

    if state != None:
        model.load_state_dict(state)

    return model

def evalModel(model, criterion, dataloader, mode="Validation"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = 0
    accuracy = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, categories in dataloader:
            images, categories = images.to(device), categories.to(device)

            outs = model.forward(images)
            loss = criterion(outs, categories)

            loss += loss.item()

            top_p, top_class = torch.exp(outs).topk(1, dim=1)
            equals = top_class == categories.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            model.train()
    loss = loss/len(dataloader)
    accuracy = accuracy/len(dataloader)
    print(f"{mode} loss: {loss:.3f}.. ")
    print(f"{mode} accuracy: {accuracy:.3f}")
    
    return loss, accuracy
            
            
def saveModel(model, arch, hidden, toDir, classToIdx):
    filepath = toDir + "/checkpoint.pth"
    checkpoint = {
        "stateDict" : model.state_dict(),
        "classToIdx": classToIdx,
        "arch" : arch,
        "hidden" : hidden
        
    }
    torch.save(checkpoint, filepath)
    
def loadModel(path):
    checkpoint = torch.load(path)
    
    model = buildModel(checkpoint["stateDict"], hidden=checkpoint["hidden"], architecture=checkpoint["arch"])
    model.class_to_idx = checkpoint["classToIdx"]
    return model