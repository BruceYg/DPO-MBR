import argparse
import os
import random

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from accelerate import Accelerator

import torch

from openicl import DatasetReader, PromptTemplate
from openicl import RandomRetriever, GenInferencer


def main(args):
    torch.manual_seed(args.exp)
    random.seed(args.exp)
    
    lmap = {'zh': 'Chinese', 'en': 'English', 'de': 'German'}

    if args.dataset == 'wmt':
        #example_list = [1480, 1312, 305, 1545, 1716, 1420, 1314, 795]

        # 20 zh-en
        #example_list = [1480, 1312]

        # 20 en-zh
        #example_list = [696, 441]

        # 20 de-en
        example_list = [445, 140]

        # 20 en-de
        #example_list = [696, 441]

        # data preparation
        src_name = f'wmt{str(args.year-1)}_{args.src_lang}-{args.tgt_lang}.txt'
        ref_name = f'wmt{str(args.year-1)}_{args.src_lang}-{args.tgt_lang}_ref.txt'

        ex_src_data = [line.strip() for i, line in enumerate(open(src_name).readlines()) if i in example_list]
        ex_ref_data = [line.strip() for i, line in enumerate(open(ref_name).readlines()) if i in example_list]
        src_data = [line.strip() for i, line in enumerate(open(src_name).readlines()) if i not in example_list]
        ref_data = [line.strip() for i, line in enumerate(open(ref_name).readlines()) if i not in example_list]


        ex_set = Dataset.from_dict({args.src_lang: ex_src_data, args.tgt_lang: ex_ref_data})
        src_set = Dataset.from_dict({args.src_lang: src_data, args.tgt_lang: ref_data})
        dataset = DatasetDict({'validation': ex_set, 'test': src_set})

    elif args.dataset == 'iwslt17':
        lmap = {'fr': 'French', 'en': 'English'}
        iwslt_dataset = load_dataset("iwslt2017", f"iwslt2017-{args.src_lang}-{args.tgt_lang}")
        val_dataset = iwslt_dataset['validation']
        test_dataset = iwslt_dataset['validation']
        val_src = [sample['translation'][args.src_lang] for sample in val_dataset]
        val_ref = [sample['translation'][args.tgt_lang] for sample in val_dataset]
        test_src = [sample['translation'][args.src_lang] for sample in test_dataset]
        test_ref = [sample['translation'][args.tgt_lang] for sample in test_dataset]
        valid_set = Dataset.from_dict({args.src_lang: val_src, args.tgt_lang: val_ref})
        test_set = Dataset.from_dict({args.src_lang: test_src, args.tgt_lang: test_ref})
        dataset = DatasetDict({'validation': valid_set, 'test': test_set})
    


    data = DatasetReader(dataset, input_columns=[args.src_lang], output_column=args.tgt_lang)

    # specify prompt format
    if args.prompt == 0:
        template = PromptTemplate(f'</E></{lmap[args.src_lang]}>=</{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 1:
        template = PromptTemplate(f'</E>{lmap[args.src_lang]}: </{lmap[args.src_lang]}> \n {lmap[args.tgt_lang]}: </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 2:
        template = PromptTemplate(f'</E></{lmap[args.src_lang]}>; </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 3:
        template = PromptTemplate(f'</E>Deutsch: </{lmap[args.src_lang]}> \n {lmap[args.tgt_lang]}: </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 4:
        template = PromptTemplate(f'</E></{lmap[args.src_lang]}>: </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 5:
        template = PromptTemplate(f'</E>Translate {lmap[args.src_lang]}: </{lmap[args.src_lang]}> into {lmap[args.tgt_lang]}: </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    
    # zero-shot for BLOOMZ
    elif args.prompt == 6:
        template = PromptTemplate(f'</E>Translate to {lmap[args.tgt_lang]}: </{lmap[args.src_lang]}>. Translation: </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 7:
        template = PromptTemplate(f'</E>What is \"</{lmap[args.src_lang]}>\" in {lmap[args.tgt_lang]}? </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 8:
        template = PromptTemplate(f'</E>Translation from {lmap[args.src_lang]} to {lmap[args.tgt_lang]}.\n</{lmap[args.src_lang]}>\n</{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 9:
        template = PromptTemplate(f'</E>Translation from {lmap[args.src_lang]} to {lmap[args.tgt_lang]}: </{lmap[args.src_lang]}>\n</{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 10:
        template = PromptTemplate(f'</E>Translate the following sentence from {lmap[args.src_lang]} to {lmap[args.tgt_lang]}.\nQ: </{lmap[args.src_lang]}>\nA: </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    else:
        raise NotImplementedError("Prompt format not implemented")


    if args.model == 'facebook/xglm-7.5B':
        model_name = 'xglm'
    elif args.model == 'bigscience/bloom-7b1':
        model_name = 'bloom'
    elif args.model == 'bigscience/bloomz-7b1':
        model_name = 'bloomz'
    elif args.model == 'bigscience/bloomz-7b1-mt':
        model_name = 'bloomzmt'

    dir_name1 = 'source'
    dir_name2 = f'{str(args.year)}_{args.src_lang}-{args.tgt_lang}'
    if args.dataset == 'iwslt17':
        dir_name2 = f'iwslt17_{args.src_lang}-{args.tgt_lang}'
    out_dir = os.path.join(args.out_dir, dir_name1, dir_name2)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.mode == 'sampling':
        temp_name = str(args.temp).split('.')
        output_name = f'{model_name}_ice{args.num_ice}_prompt{args.prompt}_sam{args.num_hyps}_temp{temp_name[0]}p{temp_name[1]}_ex{args.exp}'

        retriever = RandomRetriever(data, ice_num=args.num_ice, index_split='validation')
        inferencer = GenInferencer(model_name=args.model, tokenizer_name=args.model, num_beam=0, batch_size=1, 
                                    temp=args.temp, num_hyps=args.num_hyps, max_length=128, model_parallel=True, 
                                    output_json_filename=output_name, output_json_filepath=out_dir)
        predictions = inferencer.inference(retriever, ice_template=template)
    
    elif args.mode == 'beam':
        lp_name = str(args.lp).split('.')
        output_name = f'{model_name}_ice{len(example_list)}_prompt{args.prompt}_beam4_lp{lp_name[0]}p{lp_name[1]}_ex{args.exp}'

        retriever = RandomRetriever(data, ice_num=args.num_ice, index_split='validation')
        inferencer = GenInferencer(model_name=args.model, tokenizer_name=args.model, num_beam=4, batch_size=1,
                                   len_penalty=args.lp, max_length=128, model_parallel=True,
                                   output_json_filename=output_name, output_json_filepath=out_dir)
        predictions = inferencer.inference(retriever, ice_template=template)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', default='zh', help='name of source language')
    parser.add_argument('--tgt_lang', default='en', help='name of target language')
    parser.add_argument('--year', type=int, default=21)
    parser.add_argument('--dataset', type=str, default='wmt')
    parser.add_argument('--num_ice', type=int, default=4, help='number of in-context examples')
    parser.add_argument('--model', default='facebook/xglm-7.5B', help='name of the model')
    parser.add_argument('--prompt', type=int, default=0, help='index of prompt format')
    parser.add_argument('--bleurt', default='BLEURT-20', help='BLEURT checkpoint name')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--select', default='random')
    parser.add_argument('--mode', default='sampling', help='generation mode, either sampling or beam search')
    parser.add_argument('--num_hyps', type=int, default=8, help='number of hypotheses to generate')
    parser.add_argument('--temp', type=float, default=0.9, help='temperature for temperature sampling')
    parser.add_argument('--lp', type=float, default=1.0, help='length penalty for beam search')
    parser.add_argument('--exp', type=int, default=1, help='experiment index')
    
    parser.add_argument('--out_dir', type=str, default='./preference_data')
    args = parser.parse_args()

    main(args)