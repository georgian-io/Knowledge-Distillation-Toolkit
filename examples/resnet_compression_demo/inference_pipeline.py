import torch

class inference_pipeline:

    def __init__(self, device):
        self.device = device

    def run_inference_pipeline(self, model, data_loader):
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                X, y = data[0].to(self.device), data[1].to(self.device)
                outputs = model(X)
                predicted = torch.max(outputs["prob"], 1)[1]
                accuracy += predicted.eq(y.view_as(predicted)).sum().item()
        accuracy = accuracy / len(data_loader.dataset)
        return {"inference_result": accuracy}
