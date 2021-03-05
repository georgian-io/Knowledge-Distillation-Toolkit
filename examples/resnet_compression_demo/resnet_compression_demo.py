import sys
sys.path.append("../..")

import yaml
from collections import ChainMap

import torch
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import datasets, transforms

from inference_pipeline import inference_pipeline
from knowledge_distillation.kd_training import KnowledgeDistillationTraining

class StudentModel(ResNet):
    def __init__(self):
        super(StudentModel, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) #ResNet18
        self.conv1 = torch.nn.Conv2d(1, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)

    def forward(self, batch, temperature=1):
        logits = super(StudentModel, self).forward(batch)
        logits = logits / temperature
        prob = F.softmax(logits, dim=0)
        log_prob = F.log_softmax(logits, dim=0)
        return {"logits":logits, "prob":prob, "log_prob":log_prob}

class TeacherModel(ResNet):
    def __init__(self):
        super(TeacherModel, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) #ResNet34
        self.conv1 = torch.nn.Conv2d(1, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)

    def forward(self, batch, temperature=1):
        logits = super(TeacherModel, self).forward(batch)
        logits = logits / temperature
        prob = F.softmax(logits, dim=0)
        log_prob = F.log_softmax(logits, dim=0)
        return {"logits":logits, "prob":prob, "log_prob":log_prob}

def get_data_for_kd_training(batch):
    data = torch.cat([sample[0] for sample in batch], dim=0)
    data = data.unsqueeze(1)
    return data,

def main():
    config = yaml.load(open('./demo_config.yaml','r'), Loader=yaml.FullLoader)
    device = torch.device("cuda")

    # Create data loaders for training and validation
    transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
              ])
    train_kwargs = {'batch_size': 64, 'num_workers': 0}
    test_kwargs = {'batch_size': 1000, 'num_workers': 0}
    train_dataset = datasets.MNIST('/home/datasets/', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('/home/datasets/', train=False, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=get_data_for_kd_training, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    val_data_loaders = {"mnist_test": test_loader}

    # Create inference pipeline for validating the student model
    inference_pipeline_example = inference_pipeline(device)

    # Create student and teacher model
    student_model = StudentModel()
    teacher_model = TeacherModel()
    teacher_model.load_state_dict(torch.load("./saved_model/resnet34_teacher.pt"))

    # Train a student model with knowledge distillation and get its performance on dev set
    KD_resnet = KnowledgeDistillationTraining(train_data_loader = train_data_loader,
                                                val_data_loaders = val_data_loaders,
                                                inference_pipeline = inference_pipeline_example,
                                                student_model = student_model,
                                                teacher_model = teacher_model,
                                                num_gpu_used = config["knowledge_distillation"]["general"]["num_gpu_used"],
                                                temperature = config["knowledge_distillation"]["general"]["temperature"],
                                                final_loss_coeff_dict = config["knowledge_distillation"]["final_loss_coeff"],
                                                logging_param = ChainMap(config["knowledge_distillation"]["general"],
                                                                         config["knowledge_distillation"]["optimization"],
                                                                         config["knowledge_distillation"]["final_loss_coeff"],
                                                                         config["knowledge_distillation"]["pytorch_lightning_trainer"]),
                                                **ChainMap(config["knowledge_distillation"]["optimization"],
                                                           config["knowledge_distillation"]["pytorch_lightning_trainer"],
                                                           config["knowledge_distillation"]["comet_info"])
                                               )
    KD_resnet.start_kd_training()
    student_model = KD_resnet.get_student_model()

if __name__ == "__main__":
    main()
