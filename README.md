# Knowledge Distillation Toolkit

This toolkit allows you to compress a machine learning model using knowledge distillation. To use this toolkit, you need to provide a teacher model, a student model, data loaders for training and validation, and an inference pipeline. This toolkit is based on [PyTorch](https://pytorch.org/) and [PyTorch Lightning
](https://github.com/PyTorchLightning/pytorch-lightning), so teacher and student models need to be [PyTorch neural network modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), and data loaders need to be [PyTorch data loaders](https://pytorch.org/docs/stable/data.html).

![demo image](./demo_img.png)

# Start knowledge distillation training!
To start knowledge distillation training, you need to first instantiate the [KnowledgeDistillationTraining](https://github.com/georgian-io/Knowledge-Distillation-Toolkit/blob/f39eed6dd66f924058c9ee4b16453014efb07b75/knowledge_distillation/kd_training.py#L178) class, then call the [start_kd_training](https://github.com/georgian-io/Knowledge-Distillation-Toolkit/blob/f39eed6dd66f924058c9ee4b16453014efb07b75/knowledge_distillation/kd_training.py#L261) method.

In the table below, we show arguments that the constructor of `KnowledgeDistillationTraining` class takes in.

Argument Name | Type | Explanation | Default
--- | --- | --- | ---
`teacher_model` | `torch.nn.Module` | A teacher model. | `None`
`student_model` |`torch.nn.Module` | A student model. |`None`
`train_data_loader` | `torch.utils.data.DataLoader` | Data loader for the training data set. | `None`
`val_data_loaders` | `dict` | A dictionary which could contain multiple validation data loaders. The key should be the data loader's name and value is a data loader. Note that the data loader should be an instance of `torch.utils.data.DataLoader`. | `None`
`inference_pipeline` | `object` | A python class that returns the validation result. See [below](#How-does-inference-pipeline-work?) for more information on this class. | `None`
`num_gpu_used` | `int` | Number of GPUs used for training. | Required parameter. No default value
`max_epoch` | `int` | Number of training epochs. | Required parameter. No default value
`optimize_method` | `str` | Optimization method used to train the student model. Could be one of ["adam", "sgd", "adam_wav2vec2.0", "adam_distilBert", "adamW_distilBert"]. | Required parameter. No default value
`scheduler_method` | `str` | Learning rate scheduler. Could be one of ["", "linear_decay_with_warm_up", "cosine_anneal"]. No learning rate scheduling if setting to "". | Required parameter. No default value
`learning_rate` | `float` | Learning rate for knowledge distillation traininig. | Required parameter. No default value
`num_lr_warm_up_epoch` | `int` | Number of epochs to warm up (increase) the learning rate. Set to 0 if not warming up the learning rate. | Required parameter. No default value
`final_loss_coeff_dict` | `dict` | A dictionary which contains coefficients that should be multiplied with the loss. See below for more information. | Required parameter. No default value
`log_to_comet` | `bool` | Set to True if logging experiment results to comet.ml. If debugging, set this to False. | `False`
`comet_info_path` | `str` | Path to a txt file which contains api key, project name and work space at comet.ml. | `""`
`comet_exp_name` | `str` | Experiment name on comet.ml. | `""`
`temperature` | `int` | Temperature for calculating the knowledge distillation loss. | `1`
`seed` | `int` | Seed value for the experiment. | `32`
`track_grad_norm` | `int` | The norm to use when calculating the gradient for tracking. | `2`
`accumulate_grad_batches` | `int` | Number of gradient accumulation steps. | `1`
`accelerator` | `str`/`None` | Accelerators for PyTorch Lightning. See [here](https://pytorch-lightning.readthedocs.io/en/1.1.1/accelerators.html) for details. | `None`
`num_nodes` | `int` | Number of compute nodes. | `1`
`precision` | `int` | 16 bit or 32 bit training. See [here](https://pytorch-lightning.readthedocs.io/en/latest/amp.html) for details. | `16`
`deterministic` | `bool` | `deterministic` flag in PyTorch lightning. | `True`
`resume_from_checkpoint` | `str` | Path to a previous check point where the current experiment should resume from. | `""`
`logging_param` | `dict` | A dictionary which contains parameters that should be saved to comet.ml. | `None`

# Demo
We have provided two demos which use this toolkit and compress machine learning models.

Compress wav2vec 2.0: [this notebook](https://github.com/georgian-io/Knowledge-Distillation-Toolkit/blob/main/examples/wav2vec2_compression_demo/wav2vec2_compression_demo.ipynb)

Compress resnet: [this colab notebook](https://colab.research.google.com/drive/1r14Dp0tCmmdfS06a0EEqZaMTofdhhy-U?usp=sharing)

# How does inference pipeline work?

This toolkit uses inference pipeline to test the student model. The `inference_pipeline` class needs to implement a method `run_inference_pipeline`. The purpose of this method is to get the performance of the student model on a validation dataset. 

We walk you through how we created an inference pipeline in the code below. We pass `model` and `data_loader` to `run_inference_pipeline`. The `model` is a `student_model`, and `data_loader` is a validation data loader. You should have these two arguments in hands when you are using this toolkit, because you need them to instantiate the `KnowledgeDistillationTraining` class. Inside `run_inference_pipeline`, we take every data sample from `data_loader`, then pass it to the `model`. For every data sample, we calculate an accuracy based on the student model's prediction and ground truth. Finally, we calculate the overall `accuracy` and return it as a dictionary. In the returned dictionary, `inference_result` should match to the overall accuracy.

```
class inference_pipeline:

    def __init__(self):
        # Constructor method is optional.

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
```
The code above is just an example and you can create inference pipeline in whatever way you want. Just remember two rules:

1. The `inference_pipeline` class only needs to implement `run_inference_pipeline`. `run_inference_pipeline` tests a student model on a validation dataset.

2. `run_inference_pipeline` should return a dictionary, e.g. {"inference_result": a numerical value that measures the performance of a student model on a validation dataset.}

