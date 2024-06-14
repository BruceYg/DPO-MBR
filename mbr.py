import argparse
import os
import json
from tqdm import tqdm
from collections import defaultdict



import datasets
import evaluate
import torch
import numpy as np
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from py3langid.langid import LanguageIdentifier, MODEL_FILE



def mbr(data, metric='bleurt', mode='evaluate', probs=None):
    if metric == 'bleurt':
        from bleurt import score
        checkpoint = 'BLEURT-20'
        scorer = score.LengthBatchingBleurtScorer(checkpoint)
    else:
        raise NotImplementedError(f'metric {metric} is not implemented for MBR utility function')
    N = len(data.keys())
    out = defaultdict(lambda: defaultdict(list))
    for i, prompt in enumerate(tqdm(data.keys())):
        hyps = data[prompt]
        hyps = [hyp for hyp in hyps if hyp.strip() != '']
        hyps = [hyp.split('\n')[0] for hyp in hyps]
        scores = np.zeros((len(hyps), len(hyps)), dtype=np.float16)
        if len(hyps) == 0:
            continue
        for j, hyp in enumerate(hyps):
            scores[j][:] = scorer.score(candidates=[hyp]*len(hyps), references=hyps)
        sumed_scores = (np.sum(scores, axis=1) / len(hyps)).tolist()
        max_idx = sumed_scores.index(max(sumed_scores))
        min_idx = sumed_scores.index(min(sumed_scores))
        out[prompt]['hypotheses'].extend(hyps)
        out[prompt]['scores'].extend(sumed_scores)
        out[prompt]['best'].append(hyps[max_idx])
        out[prompt]['worst'].append(hyps[min_idx])
        out[prompt]['best_score'].append(max(sumed_scores))
        out[prompt]['worst_score'].append(min(sumed_scores))
    return out


def replacer(s, newstring, index, nofail=False):
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")
    if index < 0:
        return newstring + s
    if index > len(s):
        return s + newstring
    return s[:index] + newstring + s[index + 1:]


def main(args):

    # combine inputs
    file_list = []
    ex_idx = None
    try:
        ex_start_idx = args.data.find('ex') + 2
        ex_start = int(args.data[ex_start_idx])
    except:
        raise Exception('Name of data file not in the correct format')
    for i in range(args.num_exp):
        ex_idx = ex_start + i
        file_name = args.data
        file_name = replacer(file_name, str(ex_idx), args.data.find('ex')+2)
        file_list.append(file_name)
    data = defaultdict(list)
    for file in file_list:
        f = open(file, encoding='utf-8')
        d = json.load(f)
        if args.origin_prompt:
            for k in d.keys():
                data[d[k]['origin_prompt']].extend(d[k]['prediction'])
        else:
            for i, k in enumerate(d.keys()):
                data[i].extend(d[k]['prediction'])
        f.close()
    if args.extra is not None:
        f = open(args.extra, encoding='utf-8')
        d = json.load(f)
        for k in d.keys():
            data[d[k]['origin_prompt']].extend(d[k]['prediction'])
        f.close()
    if args.one_more:
        ex_idx += 1
        file_name = args.data
        file_name = replacer(file_name, str(ex_idx), args.data.find('ex')+2)
        f = open(file_name, encoding='utf-8')
        d = json.load(f)
        if args.origin_prompt:
            for k in d.keys():
                data[d[k]['origin_prompt']].append(d[k]['prediction'][0])
        else:
            for i, k in enumerate(d.keys()):
                data[i].append(d[k]['prediction'][0])
        f.close()
    out_data = mbr(data, args.metric, args.mode, probs)
    if args.mode == 'train':
        num_hyps = 8 * args.num_exp # assume each input file contains 8 hypotheses
        if args.one_more:
            num_hyps += 1
        out_name = args.data
        sam_idx = out_name.find('sam')
        out_name = replacer(out_name, str(num_hyps), sam_idx + 3)
        out_dir_name = os.path.split(out_name)[0]
        out_file_name = os.path.split(out_name)[1]
        out_file_name = f'mbr_{out_file_name[:-5]}_{args.exp_name}.json'
        out_name = os.path.join(out_dir_name, out_file_name)
        with open(out_name, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train for generating preference data, evaluate to test performance') # not using
    parser.add_argument('--data', type=str, help='path to a json file')
    parser.add_argument('--num_exp', type=int, default=1, help='number of json files to process')
    parser.add_argument('--extra', type=str, default=None, help='path to a file containing beam search outputs')
    parser.add_argument('--one_more', action='store_true', help='include one addtional hypothesis from the next file')

    parser.add_argument('--metric', type=str, default='bleurt', help='only support BLEURT right now')
    parser.add_argument('--accelerate', action='store_true', help='not using, mbr_fast is not accurate')
    
    parser.add_argument('--exp_name', type=str, help='add suffix to output file name')
    parser.add_argument('--origin_prompt', action='store_true', help='used for handling wmt22 zh-en. If true, store the original prompts as keys. Otherwise use indexes as keys.')

    parser.add_argument('--mbmbr', action='store_true', help='ignore for now')
    
    parser.add_argument('--out_dir', type=str, default='./preference_data')
    args = parser.parse_args()

    main(args)