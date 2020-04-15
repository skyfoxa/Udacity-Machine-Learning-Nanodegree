
import utils
import torch
from torch import optim, nn
import argparse


def train(trainDir, validDir, epochs=5, learn_rate=0.001, h=512, arch="vgg16", gpu=False, saveDir="./"):
    dataloaders, classToIdx = utils.loadData(trainDir, validDir)
    
    model = utils.buildModel(hidden=h, architecture=arch)
    
    device = torch.device("cuda" if gpu else "cpu")
    print(f"Device: {device}")


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    trainloader = dataloaders["train"]
    validloader = dataloaders["valid"]

    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 30
    
    last_valid_acc = 0.0
    min_valid_acc = 0.7
    
    print("Training started")
    
    for e in range(epochs):
        for images, categories in trainloader:
            images, categories = images.to(device), categories.to(device)
            steps += 1

            optimizer.zero_grad()

            outs = model.forward(images)
            loss = criterion(outs, categories)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"(Step {steps})")
                running_loss = 0

        else:
            print("\nValidation:")
            valid_loss, valid_acc = utils.evalModel(model, criterion, validloader)     
            
            if valid_acc > last_valid_acc and valid_acc >= min_valid_acc:
                last_valid_acc = valid_acc
                model.to(torch.device('cpu'))
                utils.saveModel(model, arch, h, saveDir, classToIdx)
                model.to(device)
                
            model.train()
    
    if last_valid_acc < min_valid_acc:
        model.to(torch.device('cpu'))
        utils.saveModel(model, arch, h, saveDir, classToIdx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train flower predictor')
    parser.add_argument('data_dir', action="store", type=str)
    parser.add_argument('--save_dir', type=str, default=".")
    parser.add_argument('--arch', type=str, default="vgg16", help='Feature selection architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    

    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    train(args.data_dir+"/train", args.data_dir+"/valid", args.epochs, args.learning_rate, args.hidden_units, args.arch, args.gpu, args.save_dir)
        
        
        
