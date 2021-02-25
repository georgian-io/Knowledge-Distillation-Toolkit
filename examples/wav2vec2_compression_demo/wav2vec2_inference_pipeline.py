import sys
sys.path.append("../..")
import time
import itertools as it

from jiwer import wer

import torch
import torch.nn.functional as F

from wav2letter.decoder import CriterionType
from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes

class W2lDecoder(object):
    def __init__(self, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = 1

        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        self.asg_transitions = None

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        encoder_input["features_only"] = True
        encoder_input["mask"] = False
        emissions, cnn_time, encoder_time, between_cnn_encoder_time, conv_time, max_pool_time, calc_prob_time = self.get_emissions(models, encoder_input)

        start_time = time.time()
        decoder_out = self.decode(emissions)
        decoder_time = time.time() - start_time

        return decoder_out, cnn_time, encoder_time, between_cnn_encoder_time, decoder_time, calc_prob_time, conv_time, max_pool_time

    def generate_emissions(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return emissions

    def get_emissions(self, model, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = model(**encoder_input)

        start_time = time.time()
        if self.criterion_type == CriterionType.CTC:
            emissions = encoder_out["log_prob"]
        emissions = emissions.transpose(0, 1).float().cpu().contiguous()
        calc_prob_time = time.time() - start_time

        return emissions, encoder_out["cnn_time"], encoder_out["encoder_time"], encoder_out["between_cnn_encoder_time"], \
               encoder_out["conv_time"], encoder_out["max_pool_time"], calc_prob_time

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))

class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = list()

        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)

        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}] for b in range(B)
        ]

def post_process_sentence(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == 'wordpiece':
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == 'letter':
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol is not None and symbol != 'none':
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence

def postprocess_features(feats, sample_rate):
    if feats.dim() == 2:
        feats = feats.mean(-1)

    assert feats.dim() == 1, feats.dim()

    with torch.no_grad():
        feats = F.layer_norm(feats, feats.shape)
    return feats

def process_batch_element(element, model, generator, target_dict, use_cuda=False, input_half=False):
    start_time = time.time()
    sample = dict()
    net_input = dict()

    feature = postprocess_features(element[0][0][0], element[1]).unsqueeze(0)
    padding_mask = torch.BoolTensor(feature.size(1)).fill_(False).unsqueeze(0)

    if use_cuda:
        net_input["source"] = feature.cuda()
        net_input["padding_mask"] = padding_mask.cuda()
    else:
        net_input["source"] = feature
        net_input["padding_mask"] = padding_mask

    if input_half:
        net_input["source"] = net_input["source"].half()

    data_load_time = time.time() - start_time

    sample["net_input"] = net_input

    with torch.no_grad():
        hypo, cnn_time, encoder_time, between_cnn_encoder_time, decoder_time, calc_prob_time, conv_time, max_pool_time = generator.generate(model, sample, prefix_tokens=None)

    start_time = time.time()
    hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu())
    prediction = post_process_sentence(hyp_pieces, 'letter')
    post_process_time = time.time() - start_time

    return prediction, cnn_time, encoder_time, between_cnn_encoder_time, decoder_time, data_load_time, post_process_time, calc_prob_time, conv_time, max_pool_time

class inference_pipeline:

    def __init__(self,
                 target_dict,
                 use_cuda,
                 input_half):

        self.generator = W2lViterbiDecoder(target_dict)
        self.target_dict = target_dict
        self.use_cuda = use_cuda
        self.input_half = input_half

    def run_inference_pipeline(self, model, data_loader):
        predictions = []
        ground_truths = []
        model.eval()

        for i, batch in enumerate(data_loader):
            prediction, _, _, _, _, _, _, _, _, _ = process_batch_element(batch, model=model, generator=self.generator, target_dict=self.target_dict, use_cuda=self.use_cuda, input_half=self.input_half)
            predictions.append(prediction)
            ground_truths.append(batch[2][0])

        wer_score = wer(ground_truths, predictions)
        return {"inference_result": wer_score}
