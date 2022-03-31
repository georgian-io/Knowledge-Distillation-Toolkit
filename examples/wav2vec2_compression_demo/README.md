# Distilling knowledge in Wav2vec 2.0
Code for "Shrinking Bigfoot: Reducing wav2vec 2.0 footprint" (https://arxiv.org/abs/2103.15760)

## Dependency
`jiwer`

`wav2letter`

`pytorch-lightning==1.0.8`

`torch==1.6`

`torchaudio==0.6`

Install `fairseq_mod` (our modified version of `fairseq`), first cd into `Knowledge-Distillation-Toolkit/utils/fairseq_mod`, then run `pip install --editable ./`

[optional] `pip install codecarbon`

## Checkpoints
12 transformer layer student wav2vec 2.0 model - https://drive.google.com/file/d/13HdB6W_Ik2JOElqz2bjRPo9Vy4C10vCq/view?usp=sharing

4 transformer layer student wav2vec 2.0 model - https://drive.google.com/file/d/1su6PKoFDSgAwZaQvYM5YkJA4Bj_mRPRN/view?usp=sharing

## Inference

Download one of the checkpoints, replace `MODEL_LOAD_PATH` with path to the downloaded checkpoint and run `wav2vec2_inference_demo.py`


## Training
Run `wav2vec2_compression_demo.py`

- To train a 12 layer student model, use the original wav2vec 2.0 model (download [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_960h.pt)) to initialize by setting `student_init_model_path` in `demo_config.yaml`

- To train a 4 layer student model, use the trained 12 layer model to initialize and use hyperparameters in `demo_config.yaml`
