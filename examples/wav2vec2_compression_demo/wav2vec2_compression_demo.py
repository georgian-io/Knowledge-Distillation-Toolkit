from collections import ChainMap

import yaml
import torch
import fairseq_mod

import sys
sys.path.append("../..")

from wav2vec2_inference_pipeline import inference_pipeline
from data_loader import LibriSpeechDataLoader
from knowledge_distillation.kd_training import KnowledgeDistillationTraining
from fairseq_mod.models.wav2vec.teacher_wav2vec2 import TeacherWav2Vec2Model
from fairseq_mod.models.wav2vec.student_wav2vec2 import StudentWav2Vec2Model

def get_proj_layer(fairseq_pretrained_model_path):
    """
    Get projection layer's weights and biases of wav2vec 2.0 pre-trained model
    """
    w2v = torch.load(fairseq_pretrained_model_path)
    return w2v["model"]["w2v_encoder.proj.weight"], w2v["model"]["w2v_encoder.proj.bias"]

if __name__ == "__main__":
    config = yaml.load(open('demo_config.yaml','r'), Loader=yaml.FullLoader)
    target_dict = fairseq_mod.data.Dictionary.load('ltr_dict.txt')

    libriSpeech_data_loader = LibriSpeechDataLoader(**config["data_loader"])
    train_data_loader = libriSpeech_data_loader.get_train_data_loader()
    val_data_loaders = libriSpeech_data_loader.get_val_data_loaders()

    inference_pipeline_example = inference_pipeline(target_dict, use_cuda=True, input_half=False)

    student_model = StudentWav2Vec2Model.create_student_model(target_dict=target_dict,
                                                              fairseq_pretrained_model_path=config["knowledge_distillation"]["general"]["fairseq_pretrained_model_path"],
                                                              **config["knowledge_distillation"]["student_model"])
    teacher_model = TeacherWav2Vec2Model.create_teacher_model(target_dict=target_dict,
                                                              fairseq_pretrained_model_path=config["knowledge_distillation"]["general"]["fairseq_pretrained_model_path"])

    proj_layer_weight, proj_layer_bias = get_proj_layer(fairseq_pretrained_model_path=config["knowledge_distillation"]["general"]["fairseq_pretrained_model_path"])
    student_model.init_proj_layer_to_decoder(proj_layer_weight, proj_layer_bias)
    teacher_model.init_proj_layer_to_decoder(proj_layer_weight, proj_layer_bias)

    KD_wav2vec2 = KnowledgeDistillationTraining(train_data_loader = train_data_loader,
                                            val_data_loaders = val_data_loaders,
                                            inference_pipeline = inference_pipeline_example,
                                            student_model = student_model,
                                            teacher_model = teacher_model,
                                            num_gpu_used = config["knowledge_distillation"]["general"]["num_gpu_used"],
                                            temperature = config["knowledge_distillation"]["general"]["temperature"],
                                            final_loss_coeff_dict = config["knowledge_distillation"]["final_loss_coeff"],
                                            logging_param = ChainMap(config["knowledge_distillation"]["general"], config["knowledge_distillation"]["optimization"],
                                                                     config["knowledge_distillation"]["final_loss_coeff"], config["knowledge_distillation"]["student_model"],
                                                                     config["knowledge_distillation"]["pytorch_lightning_trainer"]),
                                            **ChainMap(config["knowledge_distillation"]["optimization"],
                                                       config["knowledge_distillation"]["pytorch_lightning_trainer"],
                                                       config["knowledge_distillation"]["comet_info"])
                                            )
    KD_wav2vec2.start_kd_training()

    exit()