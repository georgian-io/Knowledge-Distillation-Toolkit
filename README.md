# Knowledge Distillation Toolkit

This toolkit allows you to compress a machine learning model using knowledge distillation. To use this toolkit, you need to provide a teacher model, a student model, data loaders for training and validation, and an inference pipeline. This toolkit is based on [PyTorch](https://pytorch.org/) and [PyTorch Lightning
](https://github.com/PyTorchLightning/pytorch-lightning), so teacher and student models need to be [PyTorch neural network modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), and data loaders need to be [PyTorch data loaders](https://pytorch.org/docs/stable/data.html).

![demo image](./demo_img.png)

To start knowledge distillation training, you need to first instantiate the [KnowledgeDistillationTraining](https://github.com/georgian-io/Knowledge-Distillation-Toolkit/blob/f39eed6dd66f924058c9ee4b16453014efb07b75/knowledge_distillation/kd_training.py#L178) class, then call the [start_kd_training](https://github.com/georgian-io/Knowledge-Distillation-Toolkit/blob/f39eed6dd66f924058c9ee4b16453014efb07b75/knowledge_distillation/kd_training.py#L261) method.

The constructor of `KnowledgeDistillationTraining` class takes in following arguments:

`teacher_model` (`torch.nn.Module`): A teacher model.

`student_model` (`torch.nn.Module`): A student model.

`train_data_loader` (`torch.utils.data.DataLoader`): Data loader for the training data set.

`val_data_loaders` (`dict`): A dictionary which could contain multiple validation data loaders. The key should be the data loader's name and value is a data loader. Note that the data loader should be an instance of `torch.utils.data.DataLoader`.

`inference_pipeline` (`object`): A python class that pass data samples from a validation data loader into a model for inference. It should return an inference score. See [below](#Knowledge-Distillation-Toolkit) for more information on this class.

`num_gpu_used` (`int`): Number of GPUs used for training.

`max_epoch` (`int`): Number of training epochs.

`optimize_method` (`str`): Optimization method used to train the student model. Could be one of ["adam", "sgd", "adam_wav2vec2.0", "adam_distilBert", "adamW_distilBert"].

`scheduler_method` (`str`): Learning rate scheduler. Could be one of ["", "linear_decay_with_warm_up", "cosine_anneal"]. No learning rate scheduling if setting to "".

`learning_rate` (`float`): Learning rate for knowledge distillation traininig. 

`num_lr_warm_up_epoch` (`int`): Number of epochs to warm up (increase) the learning rate. Set to 0 if not warming up the learning rate. 

`final_loss_coeff_dict` (`dict`): A dictionary which contains coefficients that should be multiplied with the loss. See below for more information.

`log_to_comet` (`bool`): Set to True if logging experiment results to comet.ml. If debugging, set this to False.

`comet_info_path` (`str`): Path to a txt file which contains api key, project name and work space at comet.ml.

`comet_exp_name` (`str`): Experiment name on comet.ml.

`temperature` (`int`): Temperature for calculating the knowledge distillation loss. Default: 1

`seed` (`int`): Seed value for the experiment. Default: 32

`track_grad_norm` (`int`): The norm to use when calculating the gradient for tracking. Default: 2

`accumulate_grad_batches` (`int`): Number of gradient accumulation steps. Default: 1

`accelerator` (`str`/`None`): Accelerators for PyTorch Lightning. See [here](https://pytorch-lightning.readthedocs.io/en/1.1.1/accelerators.html) for details. Default: `None`.

`num_nodes` (`int`): Number of compute nodes. Default: 1

`precision` (`int`). 16 bit or 32 bit training. See [here](https://pytorch-lightning.readthedocs.io/en/latest/amp.html) for details. Default 16

`deterministic` (`bool`). `deterministic` flag in PyTorch lightning. Default: True

`resume_from_checkpoint` (`str`). Path to a previous check point where the current experiment should resume from. Default: ""

`logging_param` (`dict`): A dictionary which contains parameters that should be saved to comet.ml. Default: None




