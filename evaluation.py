import argparse
import json

import evaluate
import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from accelerate import Accelerator

from openicl import DatasetReader, PromptTemplate
from openicl import RandomRetriever, GenInferencer, BleuEvaluator, TopkRetriever, BM25Retriever


def main(args):
    
    if args.dataset == 'wmt':
        lmap = {'zh': 'Chinese', 'en': 'English', 'de': 'German', 'is': 'Icelandic'}
        val_src_name = 'wmt' + str(args.ref_year) + '_' + args.src_lang + '-' + args.tgt_lang + '.txt'
        val_ref_name = 'wmt' + str(args.ref_year) + '_' + args.src_lang + '-' + args.tgt_lang + '_ref.txt'
        test_src_name = 'wmt' + str(args.year) + '_' + args.src_lang + '-' + args.tgt_lang + '.txt'
        test_ref_name = 'wmt' + str(args.year) + '_' + args.src_lang + '-' + args.tgt_lang + '_ref.txt'
        val_src = [line.strip() for line in open(val_src_name).readlines()]
        val_ref = [line.strip() for line in open(val_ref_name).readlines()]
        if args.job is None:
            test_src = [line.strip() for line in open(test_src_name).readlines()]
            test_ref = [line.strip() for line in open(test_ref_name).readlines()]
        else:
            test_src = [[line.strip() for line in open(test_src_name).readlines()][args.job*4+args.idx]]
            test_ref = [[line.strip() for line in open(test_ref_name).readlines()][args.job*4+args.idx]]

        valid_set = Dataset.from_dict({args.src_lang: val_src, args.tgt_lang: val_ref})
        test_set = Dataset.from_dict({args.src_lang: test_src, args.tgt_lang: test_ref})
        dataset = DatasetDict({'validation': valid_set, 'test': test_set})


    elif args.dataset == 'flores200':
        lmap = {'fra_Latn': 'French', 'eng_Latn': 'English', 'zho_Hans': 'Chinese'}
        flores_dataset = load_dataset('Muennighoff/flores200', f'{args.src_lang}-{args.tgt_lang}')
        dev_dataset = flores_dataset['dev']
        devtest_dataset = flores_dataset['devtest']
        src_name = f'sentence_{args.src_lang}'
        ref_name = f'sentence_{args.tgt_lang}'
        val_src = [sample[src_name] for sample in dev_dataset]
        val_ref = [sample[ref_name] for sample in dev_dataset]
        test_src = [sample[src_name] for sample in devtest_dataset]
        test_ref = [sample[ref_name] for sample in devtest_dataset]
        valid_set = Dataset.from_dict({args.src_lang: val_src, args.tgt_lang: val_ref})
        test_set = Dataset.from_dict({args.src_lang: test_src, args.tgt_lang: test_ref})
        dataset = DatasetDict({'validation': valid_set, 'test': test_set})

    elif args.dataset == 'iwslt17':
        lmap = {'fr': 'French', 'en': 'English'}
        iwslt_dataset = load_dataset("iwslt2017", f"iwslt2017-{args.src_lang}-{args.tgt_lang}")
        val_dataset = iwslt_dataset['validation']
        test_dataset = iwslt_dataset['test']
        val_src = [sample['translation'][args.src_lang] for sample in val_dataset]
        val_ref = [sample['translation'][args.tgt_lang] for sample in val_dataset]
        test_src = [sample['translation'][args.src_lang] for sample in test_dataset]
        test_ref = [sample['translation'][args.tgt_lang] for sample in test_dataset]
        valid_set = Dataset.from_dict({args.src_lang: val_src, args.tgt_lang: val_ref})
        test_set = Dataset.from_dict({args.src_lang: test_src, args.tgt_lang: test_ref})
        dataset = DatasetDict({'validation': valid_set, 'test': test_set})
        if args.year == 16:
            dataset = DatasetDict({'validation': valid_set, 'test': valid_set})



    data = DatasetReader(dataset, input_columns=[args.src_lang], output_column=args.tgt_lang)
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
    
    # zero-shot on BLOOMZ
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
    elif args.prompt == 11:
        template = PromptTemplate(f'</E>Translate from {lmap[args.src_lang]} to {lmap[args.tgt_lang]}.\n </{lmap[args.src_lang]}> </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    elif args.prompt == 12:
        template = PromptTemplate(f'</E>Translate to {lmap[args.tgt_lang]}: </{lmap[args.src_lang]}> </{lmap[args.tgt_lang]}>', {args.src_lang: f'</{lmap[args.src_lang]}>', args.tgt_lang: f'</{lmap[args.tgt_lang]}>'}, ice_token='</E>')
    else:
        template = PromptTemplate('</E>Bad: </Chinese>=</Chinese>\nGood: </Chinese>=</English>', {'en':'</English>', 'zh':'</Chinese>'}, ice_token='</E>')

    lp_name = str(args.lp).split('.')
    if args.model == 'facebook/xglm-7.5B':
        model_name = 'xglm'
    elif args.model == 'bigscience/bloom-7b1':
        model_name = 'bloom'
    elif args.model == 'bigscience/bloomz-7b1':
        model_name = 'bloomz'
    elif args.model == 'bigscience/bloomz-7b1-mt':
        model_name = 'bloomzmt'
    elif args.model == 'meta-llama/Llama-2-7b-hf':
        model_name = 'llama2hf'
    elif args.model == 'bigscience/mt0-xl':
        model_name = 'mt0xl'
    elif args.model == 'bigscience/mt0-xxl':
        model_name = 'mt0xxl'
    elif args.model == 'bigscience/mt0-xxl-mt':
        model_name = 'mt0xxlmt'
    elif args.model == 'bigscience/mt0-base':
        model_name = 'mt0base'
    if args.num_beam > 0:
        output_name = f'predictions_{model_name}_{str(args.year)}_{args.src_lang}-{args.tgt_lang}_{str(args.num_ice)}_beam{str(args.num_beam)}_prompt{str(args.prompt)}_len{lp_name[0]}p{lp_name[1]}_max128_{str(args.exp)}'
    elif args.num_beam == 0:
        output_name = f'predictions_{model_name}_{str(args.year)}_{args.src_lang}-{args.tgt_lang}_{str(args.num_ice)}_beam{str(args.num_beam)}_prompt{str(args.prompt)}_sam{str(args.num_hyps)}_tempp{str(args.temp)[2:]}_max128_ex{str(args.exp)}'
    if args.test == True:
        output_name = f'predictions_{model_name}_{str(args.year)}_{args.src_lang}-{args.tgt_lang}_{str(args.num_ice)}_beam{str(args.num_beam)}_prompt{str(args.prompt)}_test'
    if args.select == 'random':
        retriever = RandomRetriever(data, ice_num=args.num_ice, index_split='validation')
    elif args.select == 'topk':
        retriever = TopkRetriever(data, ice_num=args.num_ice, index_split='validation')
    elif args.select == 'bm25':
        retriever = BM25Retriever(data, ice_num=args.num_ice, index_split='validation')
    output_name += f'_{args.select}'
    output_name += '_fixed'
    if args.ckpt is not None:
        output_name += '_finetuned'
    output_name += f'_{args.name}'
    out_dir = f'./preference_data/evaluate/{args.year}_{args.src_lang}-{args.tgt_lang}'
    if args.dataset == 'flores200':
        out_dir = f'./preference_data/evaluate/flores200_{args.src_lang}-{args.tgt_lang}'
    elif args.dataset == 'iwslt17':
        out_dir = f'./preference_data/evaluate/iwslt17_{args.src_lang}-{args.tgt_lang}'
    if args.job is not None:
        output_name += f'_{str(args.job)}-{str(args.idx)}'
    if args.ckpt is not None:
        inferencer = GenInferencer(model_name=args.model, tokenizer_name=args.model, model_ckpt=args.ckpt, num_beam=args.num_beam, len_penalty=args.lp, batch_size=1, temp=args.temp, num_hyps=args.num_hyps, max_length=128, output_json_filename=output_name, model_parallel=True, output_json_filepath=out_dir)
    else:
        inferencer = GenInferencer(model_name=args.model, tokenizer_name=args.model, num_beam=args.num_beam, len_penalty=args.lp, batch_size=1, temp=args.temp, num_hyps=args.num_hyps, max_length=128, output_json_filename=output_name, model_parallel=True, output_json_filepath=out_dir)
    predictions = inferencer.inference(retriever, ice_template=template)

    if args.scoring: # only supports beam search and greedy decoding
        print('Calculating BLEU score\n')

        sacrebleu = evaluate.load("sacrebleu")
        if args.tgt_lang == 'zh':
            bleu_results = sacrebleu.compute(predictions=predictions, references=test_ref, tokenize='zh')
        else:
            bleu_results = sacrebleu.compute(predictions=predictions, references=test_ref)
        print(f'BLEU results: {bleu_results}')

        print('\nCalculating COMET scores\n')
        comet_metric20 = evaluate.load('comet', config_name='Unbabel/wmt20-comet-da', cache_dir='.cache')
        results = comet_metric20.compute(predictions=predictions, references=test_ref, sources=test_src)
        comet_mean = results['mean_score']
        print(f'COMET20: {comet_mean}')
        del comet_metric20

        comet_metric22 = evaluate.load('comet', config_name='Unbabel/wmt22-comet-da', cache_dir='.cache')
        results = comet_metric22.compute(predictions=predictions, references=test_ref, sources=test_src)
        comet_mean = results['mean_score']
        print(f'COMET22: {comet_mean}')
        del comet_metric22

        print('\nCalculating BLEURT score\n')

        from bleurt import score
        checkpoint = "BLEURT-20"
        scorer = score.BleurtScorer(checkpoint)
        scores = scorer.score(references=test_ref, candidates=predictions)
        bleurt_score = sum(scores) / len(scores)
        print(f'BLEURT results: {bleurt_score}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', default='zh', help='name of source language')
    parser.add_argument('--tgt_lang', default='en', help='name of target language')
    parser.add_argument('--dataset', type=str, default='wmt')
    parser.add_argument('--year', type=int, default=21)
    parser.add_argument('--ref_year', type=int, default=20)
    parser.add_argument('--num_ice', type=int, default=4, help='number of in-context examples')
    parser.add_argument('--num_beam', type=int, default=4, help='beam width for decoding, 0 for sampling')
    parser.add_argument('--lp', type=float, default=1.0, help='length penalty for beam search')
    parser.add_argument('--model', default='facebook/xglm-7.5B', help='name of the model')
    parser.add_argument('--ckpt', type=str, default=None, help='name of the checkpoint file. By default, the checkpoint is saved in .cache/gy266')
    parser.add_argument('--prompt', type=int, default=0, help='index of prompt format')
    parser.add_argument('--bleurt', default='BLEURT-20', help='BLEURT checkpoint name')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--select', default='random')
    parser.add_argument('--num_hyps', type=int, default=8, help='number of hypotheses to generate')
    parser.add_argument('--temp', type=float, default=0.9, help='temperature for temperature sampling')
    parser.add_argument('--exp', type=int, default=1, help='experiment index')
    parser.add_argument('--job', default=None, type=int)
    parser.add_argument('--idx', default=None, type=int)
    parser.add_argument('--name', type=str)
    parser.add_argument('--scoring', action='store_true')
    args = parser.parse_args()

    main(args)


    