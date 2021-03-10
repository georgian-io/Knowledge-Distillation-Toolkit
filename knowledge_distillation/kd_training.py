import os

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import numpy as np
from tqdm import tqdm


class KnowledgeDistillWav2vec2Model(pl.LightningModule):
    """
    Distill the knowledge from a teacher model to a student model.

    Args:
        num_gpu_used (int): number of GPUs used for training
        max_epoch (int): number of training eoochs
        temperature (int): temperature for knowledge distillation
        optimize_method (str):specify the optimization method used to train the student model
        scheduler_method (str, optional): specify how learning rate could be scheduled
        learning_rate (float): learning rate for the optimization method
        num_lr_warm_up_epoch (int): number of epochs for learning rate warm up
        final_loss_coeff_dict (dict): a dictionary which contains coefficients that will be multipled with different loss values, e.g. knowledge distillation loss, student loss ,etc.
        train_data_loader (torch.utils.data.DataLoader): data loader for the training data set
        val_data_loaders (dict): data loaders for validation
        inference_pipeline (object): an inference pipeline which runs a model on a given validation dataset.
        student_model (torch.nn.Module): the student model
        teacher_model (torch.nn.Module): the teacher model
    """

    def __init__(self,
                 num_gpu_used,
                 max_epoch,
                 temperature,
                 optimize_method,
                 scheduler_method,
                 learning_rate,
                 num_lr_warm_up_epoch,
                 final_loss_coeff_dict,
                 train_data_loader,
                 val_data_loaders,
                 inference_pipeline,
                 student_model,
                 teacher_model,
        ):
        super().__init__()

        self.optimize_method = optimize_method
        self.scheduler_method = scheduler_method
        self.final_loss_coeff_dict = final_loss_coeff_dict

        # Set hyper parameters
        if num_gpu_used == 1:
            self.max_epoch = max_epoch
            self.lr = learning_rate
            self.num_lr_warm_up_epoch = num_lr_warm_up_epoch
            self.temperature = temperature
        else:
            self.register_buffer("max_epoch", torch.tensor(max_epoch))
            self.register_buffer("lr", torch.tensor(learning_rate))
            self.register_buffer("num_lr_warm_up_epoch", torch.tensor(num_lr_warm_up_epoch))
            self.register_buffer("temperature", torch.tensor(temperature))

        # Set data loaders
        self.train_data_loader = train_data_loader
        self.val_data_loaders = val_data_loaders

        # Set inference pipeline
        self.inference_pipeline = inference_pipeline

        # Set student and teacher models
        self.student_model = student_model
        self.teacher_model = teacher_model


    def training_step(self, batch, batch_idx):
        final_loss, logit_diff, prob_diff, final_loss_components = self(batch)

        for loss_name, loss_val in final_loss_components.items():
            self.log(loss_name, loss_val, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_final_loss', final_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_logit_diff', logit_diff, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_prob_diff', prob_diff, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return final_loss

    def forward(self, batch):
        """
        One forward pass of knowledge distillation training

        """

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_net_output = self.teacher_model(temperature=self.temperature, *batch)
        self.student_model.train()
        student_net_output = self.student_model(temperature=self.temperature, *batch)

        teacher_prob = teacher_net_output["prob"]
        teacher_logits = teacher_net_output["logits"]
        student_prob = student_net_output["prob"]
        student_logits = student_net_output["logits"]
        student_log_prob = student_net_output["log_prob"]

        final_loss_components = student_net_output["loss"] if "loss" in student_net_output else dict()
        final_loss_components["kd_loss"] = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (self.temperature**2)
        final_loss_components["cos_embed_loss"] = 1 - torch.mean(F.cosine_similarity(student_logits, teacher_logits, dim=-1)) #this loss will be small if two tensor are similar

        logit_diff = F.mse_loss(student_logits, teacher_logits)
        prob_diff = F.l1_loss(student_prob, teacher_prob)

        final_loss = 0
        for name_of_loss, coeff in self.final_loss_coeff_dict.items():
            if name_of_loss in final_loss_components: final_loss = final_loss + coeff*final_loss_components[name_of_loss]

        return final_loss, logit_diff, prob_diff, final_loss_components

    def validation_step(self, batch, batch_idx):
        return

    def validation_epoch_end(self, outputs):
        self.student_model.eval()

        for val_data_loader_name, val_data_loader in self.val_data_loaders.items():
            val_result = self.inference_pipeline.run_inference_pipeline(self.student_model, val_data_loader)
            self.log(val_data_loader_name, val_result["inference_result"], on_epoch=True, prog_bar=True, logger=True)
            print("\n" + val_data_loader_name + " :" +  str(val_result["inference_result"]) + "\n")

        torch.cuda.empty_cache()
        print("GPU " + str(torch.cuda.current_device()) + " current active MB: " + str(torch.cuda.memory_stats()["active_bytes.all.current"] * 1e-6))
        print("GPU " + str(torch.cuda.current_device()) + " current reserved MB: " + str(torch.cuda.memory_stats()["reserved_bytes.all.current"] * 1e-6))
        return

    def configure_optimizers(self):
        if self.optimize_method == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimize_method == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimize_method == "adam_wav2vec2.0":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-6) # wav2vec2,0's optimizer set up on Adam. (Need to verify)
        elif self.optimize_method == "adam_distilBert":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-6) # distilBert's optimzer setup on Adam
        elif self.optimize_method == "adamW_distilBert":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-6)
        else:
            raise Exception("Could not find optimizer")

        if self.scheduler_method == "":
            return optimizer
        elif self.scheduler_method == "linear_decay_with_warm_up":
            def lr_lambda(current_epoch): # Copied from https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
                if current_epoch < self.num_lr_warm_up_epoch:
                    return float(current_epoch+1) / float(max(1, self.num_lr_warm_up_epoch)) # current_epoch+1 to prevent lr=0 in epoch 0
                return max(
                    0.0, float(self.max_epoch - current_epoch) / float(max(1, self.max_epoch - self.num_lr_warm_up_epoch)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.scheduler_method == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=9, T_mult=1, eta_min=1e-6)
        else:
            raise Exception("Could not find scheduler")

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return list(self.val_data_loaders)[0]

    def get_student_model(self):
        return self.student_model


class KnowledgeDistillationTraining():

    def __init__(self,
                 num_gpu_used,
                 max_epoch,
                 optimize_method,
                 scheduler_method,
                 learning_rate,
                 num_lr_warm_up_epoch,
                 final_loss_coeff_dict,
                 log_to_comet = False,
                 comet_info_path = "",
                 comet_exp_name = "",
                 temperature = 1,
                 seed = 32,
                 track_grad_norm = 2,
                 accumulate_grad_batches = 1,
                 accelerator = None,
                 num_nodes = 1,
                 precision = 16,
                 deterministic = True,
                 resume_from_checkpoint = None,
                 train_data_loader = None,
                 val_data_loaders = None,
                 inference_pipeline = None,
                 student_model = None,
                 teacher_model = None,
                 logging_param = None,
        ):
        seed_everything(seed)

        checkpoint_callback = ModelCheckpoint(monitor='train_final_loss',
                                              filename='student-{epoch:03d}-{train_final_loss:.5f}',
                                              save_top_k=3,
                                              mode='min')

        lr_monitor_callback = LearningRateMonitor(logging_interval='step')

        if log_to_comet:
            comet_info = open(comet_info_path, 'r')
            comet_api_key = comet_info.readline().strip('\n')
            comet_proj_name = comet_info.readline().strip('\n')
            comet_workspace = comet_info.readline().strip('\n')
            comet_logger = CometLogger(api_key = comet_api_key,
                                       workspace = comet_workspace,
                                       save_dir = '.',
                                       project_name = comet_proj_name,
                                       experiment_name = comet_exp_name,
                                       auto_histogram_weight_logging = True)
            comet_info.close()
            comet_logger.log_hyperparams(dict(logging_param))

        self.trainer = Trainer(max_epochs = max_epoch,
                               track_grad_norm = track_grad_norm,
                               accumulate_grad_batches = accumulate_grad_batches,
                               gpus = num_gpu_used,
                               accelerator = "ddp" if num_gpu_used>1 else None,
                               num_nodes = num_nodes,
                               precision = precision,
                               logger = comet_logger if log_to_comet else None,
                               deterministic = deterministic,
                               resume_from_checkpoint = None if resume_from_checkpoint == "" else resume_from_checkpoint,
                               callbacks = [checkpoint_callback, lr_monitor_callback] if log_to_comet else [],
                               #replace_sampler_ddp = False,
                               #accelerator = "ddp_cpu",
                               #num_processes = 2
                        )

        self.pytorch_lightning_module = KnowledgeDistillWav2vec2Model(num_gpu_used = num_gpu_used,
                                                                      max_epoch = max_epoch,
                                                                      temperature = temperature,
                                                                      optimize_method = optimize_method,
                                                                      scheduler_method = scheduler_method,
                                                                      learning_rate = learning_rate,
                                                                      num_lr_warm_up_epoch = num_lr_warm_up_epoch,
                                                                      final_loss_coeff_dict = final_loss_coeff_dict,
                                                                      train_data_loader = train_data_loader,
                                                                      val_data_loaders = val_data_loaders,
                                                                      inference_pipeline = inference_pipeline,
                                                                      student_model = student_model,
                                                                      teacher_model = teacher_model,
                                                                      )

    def start_kd_training(self):
        self.trainer.fit(self.pytorch_lightning_module)

    def get_student_model(self):
        return self.pytorch_lightning_module.get_student_model()
