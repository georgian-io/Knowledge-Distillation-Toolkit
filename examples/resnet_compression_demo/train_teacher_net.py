from tqdm import tqdm

import torch
from torch import nn, optim

import torchvision
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import datasets, transforms

from inference_pipeline import inference_pipeline

class TeacherModel(ResNet):
    def __init__(self):
        super(TeacherModel, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) #ResNet34
        self.conv1 = torch.nn.Conv2d(1, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)

def train(model, train_loader, optimizer, loss_function, device):
    total_loss = 0
    model.train()
    progress = tqdm(enumerate(train_loader), desc="Train Loss: ", total=len(train_loader))
    for i, data in progress:
        X, y = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        total_loss += current_loss
        progress.set_description("Train Loss: {:.4f}".format(total_loss/(i+1)))
    return model

def main():
    total_epoch = 5
    device = torch.device("cuda")

    transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
              ])
    train_kwargs = {'batch_size': 64, 'num_workers': 4}
    test_kwargs = {'batch_size': 1000, 'num_workers': 4}
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = TeacherModel().to(device)
    optimizer = optim.Adadelta(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    inference_pipeline_example = inference_pipeline(device)

    for epoch in range(total_epoch):
        model = train(model, train_loader, optimizer, loss_function, device)
        result = inference_pipeline_example.run_inference_pipeline(model, test_loader)
        val_acc = result["inference_result"]
        print(f"epoch {epoch}, validation accuracy = {val_acc} \n")
    torch.save(model.state_dict(), "./saved_model/resnet34_teacher.pt")


if __name__ == "__main__":
    main()



