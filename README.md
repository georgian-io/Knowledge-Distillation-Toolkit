# Knowledge Distillation Toolkit

This toolkit allows you to compress a machine learning model using knowledge distillation. To use this toolkit, you need to provide a teacher model, a student model, data loaders for training and validation, and an inference pipeline. This toolkit is based on PyTorch and [PyTorch Lightning
](https://github.com/PyTorchLightning/pytorch-lightning), so teacher and student models need to be subclasses of [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), and data loaders need to be [PyTorch data loaders](https://pytorch.org/docs/stable/data.html).
