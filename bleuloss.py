from torch.cuda import LongTensor, FloatTensor
import torch
from torch import nn
import torch.nn.functional as F


def batch_log_bleulosscnn_ae(decoder_outputs, target_idx, ngram_list, trans_len=None, pad=0, weight_list=None):
    """
    decoder_outputs: [output_len, batch_size, vocab_size]
        - matrix with probabilityes  -- log probs
    target_variable: [batch_size, target_len]
        - reference batch
    ngram_list: int or List[int]
        - n-gram to consider
    pad: int
        the idx of "pad" token
    weight_list : List
        corresponding weight of ngram

    NOTE: output_len == target_len
    """
    batch_size, output_len, vocab_size = decoder_outputs.size()
    _, tgt_len = target_idx.size()
    if type(ngram_list) == int:
        ngram_list = [ngram_list]
    if ngram_list[0] <= 0:
        ngram_list[0] = output_len
    if weight_list is None:
        weight_list = [1. / len(ngram_list)] * len(ngram_list)
    decoder_outputs = torch.log_softmax(decoder_outputs,dim=-1)
    decoder_outputs = torch.relu(decoder_outputs + 20) - 20
    index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)
    cost_nll = decoder_outputs.gather(dim=2, index=index)
    cost_nll = cost_nll.unsqueeze(1)
    out = cost_nll
    sum_gram = 0. #FloatTensor([0.])
    zero = torch.tensor(0.0).cuda()
    target_expand = target_idx.view(batch_size,1,1,-1).expand(-1,-1,output_len,-1)
    out = torch.where(target_expand==pad, zero, out)
    for cnt, ngram in enumerate(ngram_list):
        if ngram > output_len:
            continue
        term = out.squeeze(0)
        if ngram < decoder_outputs.size()[1]:
            term = term.squeeze(1)
            gum_tmp = F.gumbel_softmax(term, tau=1, dim=1)
            term = term.mul(gum_tmp).sum(1).mean(1)
        else:
            while len(term.shape) > 1:
                assert term.shape[-1] == 1, str(term.shape)
                term = term.sum(-1)
        try:
            sum_gram += weight_list[cnt] * term
        except:
            print(sum_gram.shape)
            print(term.shape)
            print((weight_list[cnt] * term).shape)
            print(ngram)
            print(decoder_outputs.size()[1])
            assert False

    loss = -sum_gram
    return loss

def my_bleulosscnn_ae(decoder_outputs, target_idx, ngram_list, trans_len=None, pad=0, weight_list=None):
    """
    decoder_outputs: [output_len, batch_size, vocab_size]
        - matrix with probabilityes  -- log probs
    target_variable: [batch_size, target_len]
        - reference batch
    ngram_list: int or List[int]
        - n-gram to consider
    pad: int
        the idx of "pad" token
    weight_list : List
        corresponding weight of ngram

    NOTE: output_len == target_len
    """
    # print(decoder_outputs.shape)
    batch_size, output_len, vocab_size = decoder_outputs.size()
    _, tgt_len = target_idx.size()
    if type(ngram_list) == int:
        ngram_list = [ngram_list]
    if ngram_list[0] <= 0:
        ngram_list[0] = output_len
    if weight_list is None:
        weight_list = [1. / len(ngram_list)] * len(ngram_list)
    decoder_outputs = torch.softmax(decoder_outputs, dim=-1)
    index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)
    cost_nll = decoder_outputs.gather(dim=2, index=index)
    cost_nll = cost_nll.unsqueeze(1)
    # exit()
    loss = torch.relu(0.9 - cost_nll.sum(-2)).mean()
    return loss

def my_bleuloss_STG(decoder_outputs, target_idx, ngram_list, trans_len=None, pad=0, weight_list=None):
    """
    decoder_outputs: [output_len, batch_size, vocab_size]
        - matrix with probabilityes  -- log probs
    target_variable: [batch_size, target_len]
        - reference batch
    ngram_list: int or List[int]
        - n-gram to consider
    pad: int
        the idx of "pad" token
    weight_list : List
        corresponding weight of ngram

    NOTE: output_len == target_len
    """
    batch_size, output_len, vocab_size = decoder_outputs.size()
    _, tgt_len = target_idx.size()
    decoder_outputs = torch.softmax(decoder_outputs, dim=-1)
    index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)
    onehot_prediction = F.one_hot(torch.argmax(decoder_outputs, dim=-1), num_classes=vocab_size)
    soft_onehot_prediction = onehot_prediction + decoder_outputs - decoder_outputs.detach()

    targe_onehot = F.one_hot(index, num_classes=vocab_size)
    soft_onehot_prediction = soft_onehot_prediction.unsqueeze(-2).repeat(1, 1, tgt_len, 1)

    diff = 2 - (targe_onehot - soft_onehot_prediction).abs().sum(-1)
    loss = (diff.sum(1)-2).abs().mean()

    return loss