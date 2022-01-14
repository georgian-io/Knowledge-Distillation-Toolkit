# Distilling knowledge in Wav2vec 2.0
Code for "Shrinking Bigfoot: Reducing wav2vec 2.0 footprint" (https://arxiv.org/abs/2103.15760)

## Dependency
`jiwer`

`wav2letter`

`pytorch-lightning`

`torch==1.6`

`torchaudio==0.6`

Install `fairseq_mod` (our modified version of `fairseq`), first cd into `Knowledge-Distillation-Toolkit/utils/fairseq_mod`, then run `pip install --editable ./`

## Checkpoints
12 layer student wav2vec 2.0 model - 

4 layer student wav2vec 2.0 model - 

## Inference
`wav2vec2_inference_demo.py`

## Training
`wav2vec2_compression_demo.py`
