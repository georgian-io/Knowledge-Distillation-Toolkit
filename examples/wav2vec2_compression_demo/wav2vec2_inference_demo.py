import yaml
import torch
import fairseq_mod

import sys
sys.path.append("../..")

from wav2vec2_inference_pipeline import inference_pipeline
from wav2vec2_compression_demo import get_proj_layer
from data_loader import LibriSpeechDataLoader
from knowledge_distillation.kd_training import KnowledgeDistillModel
from fairseq_mod.models.wav2vec.student_wav2vec2 import StudentWav2Vec2Model
from fairseq_mod.models.wav2vec.teacher_wav2vec2 import TeacherWav2Vec2Model

from codecarbon import EmissionsTracker

if __name__ == "__main__":
    MODEL_LOAD_PATH = "/home/Knowledge-Distillation-Toolkit/examples/wav2vec2_compression_demo/speech-processing/retrain_exp9/checkpoints/student-epoch=042-train_final_loss=0.08857.ckpt"
    config = yaml.load(open('demo_config.yaml','r'), Loader=yaml.FullLoader)
    target_dict = fairseq_mod.data.Dictionary.load('ltr_dict.txt')

    libriSpeech_data_loader = LibriSpeechDataLoader(**config["data_loader"])
    val_data_loaders = libriSpeech_data_loader.get_val_data_loaders()

    inference_pipeline_example = inference_pipeline(target_dict, use_cuda=True, input_half=False)

    student_model = StudentWav2Vec2Model.create_student_model(target_dict=target_dict,
                                                              fairseq_pretrained_model_path=config["knowledge_distillation"]["general"]["fairseq_pretrained_model_path"],
                                                              **config["knowledge_distillation"]["student_model"])
    teacher_model = TeacherWav2Vec2Model.create_teacher_model(target_dict=target_dict,
                                                              fairseq_pretrained_model_path=config["knowledge_distillation"]["general"]["fairseq_pretrained_model_path"])
    proj_layer_weight, proj_layer_bias = get_proj_layer(fairseq_pretrained_model_path=config["knowledge_distillation"]["general"]["fairseq_pretrained_model_path"])
    teacher_model.init_proj_layer_to_decoder(proj_layer_weight, proj_layer_bias)
    student_model.init_proj_layer_to_decoder(torch.nn.Parameter(torch.rand(proj_layer_weight.shape)), torch.nn.Parameter(torch.rand(proj_layer_bias.shape)))

    KD_module = KnowledgeDistillModel.load_from_checkpoint(
        MODEL_LOAD_PATH,
        num_gpu_used = None,
        max_epoch = 0,
        temperature = 0,
        optimize_method = None,
        scheduler_method = None,
        learning_rate = 0,
        num_lr_warm_up_epoch = 0,
        final_loss_coeff_dict = None,
        train_data_loader = None,
        val_data_loaders = None,
        inference_pipeline = None,
        student_model = student_model,
        teacher_model = teacher_model)
    KD_module.student_model.cuda()
    KD_module.student_model.eval()
    tracker = EmissionsTracker()
    tracker.start()
    print(inference_pipeline_example.run_inference_pipeline(KD_module.student_model, val_data_loaders['dev_clean']))
    tracker.stop()
    exit()