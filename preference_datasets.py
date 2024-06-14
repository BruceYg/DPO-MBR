import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import json
import random
import itertools
from operator import itemgetter
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_max_diff_pairs(scores, n=5):
    indexes = list(range(len(scores)))
    pairs = list(itertools.permutations(scores, 2))
    idx_pairs = list(itertools.permutations(indexes, 2))
    diff_pairs = [(pair, pair[0] - pair[1], i) for i, pair in enumerate(pairs)]
    diff_pairs.sort(key=itemgetter(1), reverse=True)
    return [idx_pairs[p[2]] for p in diff_pairs[:n]]


def get_beam(data, n=3):
    pass


def get_single_pair(data):
    """
    (best, worst)
    """
    out = defaultdict(dict)
    for k in data.keys():
        winner = data[k]['best'][0]
        loser = data[k]['worst'][0]
        responses = [winner, loser]
        pairs = [(0, 1)]
        out[k]['responses'] = responses
        out[k]['pairs'] = pairs
        out[k]['sft_target'] = data[k]['best'][0]
    return out


def get_double_pair(data):
    out = defaultdict(dict)
    for k in data.keys():
        responses = data[k]['hypotheses']
        scores = data[k]['scores']
        pairs = []
        zipped_pairs = zip(scores, responses)
        sorted_pairs = sorted(zipped_pairs, reverse=True)
        responses = [res for _, res in sorted_pairs]
        pairs.append((0, len(responses)-1))
        pairs.append((1, len(responses)-2))
        out[k]['responses'] = responses
        out[k]['pairs'] = pairs
        out[k]['sft_target'] = data[k]['best'][0]
    return out
 


def get_beam_pairs(data, mode='3pairs'):
    """
    2pairs: (best, beam), (beam, worst)
    3pairs: (best, beam), (beam, worst), (best, worst)
    all: (y1, beam), (y2, beam), ...
    """
    out = defaultdict(dict)
    for k in data.keys():
        responses = data[k]['hypotheses']
        scores = data[k]['scores']
        best_score = data[k]['best_score'][0]
        worst_score = data[k]['worst_score'][0]
        best_idx = scores.index(best_score)
        worst_idx = scores.index(worst_score)
        beam_idx = len(scores) - 1
        pairs = []
        if mode != 'all':
            if best_idx != beam_idx:
                pairs.append((best_idx, beam_idx))
            if worst_idx != beam_idx:
                pairs.append((beam_idx, worst_idx))
            if best_idx != beam_idx and worst_idx != beam_idx and mode == '3pairs':
                pairs.append((best_idx, worst_idx))
        elif mode == 'all':
            beam_score = scores[-1]
            for idx, score in enumerate(scores[:-1]):
                if score > beam_score:
                    pairs.append((idx, beam_idx))
                elif score < beam_score:
                    pairs.append((beam_idx, idx))
                else:
                    pass
        else:
            raise NotImplementedError(f'{mode} not implemented')
        out[k]['responses'] = responses
        out[k]['pairs'] = pairs
        out[k]['sft_target'] = data[k]['best'][0]
    return out


def get_anchor_pair(data, anchor_idx=3, num_hyps=8):
    out = defaultdict(dict)
    for k in data.keys():
        responses = data[k]['hypotheses']
        if len(responses) < num_hyps:
            continue
        scores = data[k]['scores']
        pairs = []
        zipped_pairs = zip(scores, responses)
        sorted_pairs = sorted(zipped_pairs, reverse=True)
        responses = [res for _, res in sorted_pairs]
        # for i in range(len(responses)):
        #     if i < anchor_idx:
        #         pairs.append((i, anchor_idx))
        #     elif i > anchor_idx:
        #         pairs.append((anchor_idx, i))
        pairs.append((0, anchor_idx))
        pairs.append((anchor_idx, len(responses)-1))
        out[k]['responses'] = responses
        out[k]['pairs'] = pairs
        out[k]['sft_target'] = data[k]['best'][0]
    return out


def get_con_pairs(data, stride=1):
    out = defaultdict(dict)
    for k in data.keys():
        responses = data[k]['hypotheses']
        #if stride >= len(responses):
        #    raise 
        scores = data[k]['scores']
        pairs = []
        zipped_pairs = zip(scores, responses)
        sorted_pairs = sorted(zipped_pairs, reverse=True)
        responses = [res for _, res in sorted_pairs]
        for i in range(0, len(responses) - stride, stride):
            pairs.append((i, i+stride))
        out[k]['responses'] = responses
        out[k]['pairs'] = pairs
        out[k]['sft_target'] = data[k]['best'][0]
    return out


def get_ranking_cheat(data):
    out = defaultdict(dict)
    for k in data.keys():
        responses = data[k]['hypotheses']
        scores = data[k]['scores']
        pairs = []
        zipped_pairs = zip(scores, responses)
        sorted_pairs = sorted(zipped_pairs, reverse=True)
        responses = [res for _, res in sorted_pairs]
        responses = responses[::2]
        # if len(responses) % 2 != 0:
        #     responses.append(responses[-1])
        for i in range(0, len(responses)//2, 1):
            # chosen: (y_1,y_2,y_3,y_4), rejected: (y_5,y_6,y_7,y_8)
            pairs.append((i, i+len(responses)//2-1))
        out[k]['responses'] = responses
        out[k]['pairs'] = pairs
        out[k]['sft_target'] = data[k]['best'][0]
    return out


def get_wmt(split, mode: str = 'single', num_hyps: int = 8, with18: bool = False, lang='zhen', silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    inputs = []
    winners = []
    losers = []
    f2 = None
    f3 = None

    data = defaultdict(dict)
    if split == 'train':
        if mode == 'single' or mode == 'double' or mode == 'anchor2' or mode == 'rank':
            f = open('preference_data/source/20_zh-en/mbr_bloomzmt_ice2_prompt0_sam8_temp0p7_ex0_baseNormMBR2.json')
        elif mode == 'consec_noBeam':
            f = open('preference_data/source/20_zh-en/mbr_bloomzmt_ice2_prompt0_sam9_temp0p7_ex0_baseNormMBR2.json')
        else:
            f = open('preference_data/source/20_zh-en/mbr_bloomzmt_ice2_prompt0_sam8_temp0p7_ex0_withBeamNormMBR2.json')
    elif split == 'test':
        f = open('preference_data/source/18_zh-en/mbr_bloomzmt_ice2_prompt0_sam8_temp0p7_ex0_baseNormMBR2.json')
    input_data = json.load(f)

    if mode == 'single':
        data = get_single_pair(input_data)
    elif mode == 'double':
        data = get_double_pair(input_data)
    elif mode == 'withBeam2':
        data = get_beam_pairs(input_data, mode='2pairs')
    elif mode == 'withBeam3':
        data = get_beam_pairs(input_data, mode='3pairs')
    elif mode == 'withBeamH':
        data = get_beam_pairs(input_data, mode='all')
    elif mode == 'consec' or mode == 'consec_noBeam':
        data = get_con_pairs(input_data, stride=1)
    elif mode == 'anchor2':
        anchor_idx = num_hyps // 2 - 1
        data = get_anchor_pair(input_data, anchor_idx=anchor_idx, num_hyps=num_hyps)
    elif mode == 'rank':
        data = get_ranking_cheat(input_data)
    return data


def get_flores(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    dataset = datasets.load_dataset('Muennighoff/flores200', 'zho_Hans-eng_Latn')
    if split == 'train':
        dataset = dataset.shuffle(seed=42)['dev']
    elif split == 'test':
        dataset = dataset.shuffle(seed=42)['devtest']

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, disable=silent):
        source_sentence = row['sentence_zho_Hans']
        target_sentence = row['sentence_eng_Latn']
        prompt = f'Translate the following sentence from Chinese to English.\nQ: {source_sentence}\nA: '
        data[prompt]['responses'] = [target_sentence]
        data[prompt]['pairs'] = [(0,0)]
        data[prompt]['sft_target'] = target_sentence
    
    return data


def get_wmt_sft(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(dict)
    if split == 'train':
        years = ['17', '18', '19', '20']
    elif split == 'test':
        years = ['21']
    srcs = []
    refs = []
    for year in years:
        src_file = open(f'wmt{year}_en-zh.txt')
        ref_file = open(f'wmt{year}_en-zh_ref.txt')
        src_sentences = [line.strip() for line in src_file.readlines()]
        ref_sentences = [line.strip() for line in ref_file.readlines()]
        srcs += src_sentences
        refs += ref_sentences
    for i, s in enumerate(srcs):
        r = refs[i]
        prompt = f'Translate from English to Chinese.\n{s}'
        data[prompt]['responses'] = [r]
        data[prompt]['pairs'] = [(0,0)]
        data[prompt]['sft_target'] = r
    return data

def get_iwslt_sft(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(dict)
    if split == 'train':
        iwslt_dataset = datasets.load_dataset("iwslt2017", f"iwslt2017-en-fr")['train']
        src = [sample['translation']['en'] for sample in iwslt_dataset][:20000]
        ref = [sample['translation']['fr'] for sample in iwslt_dataset][:20000]
    elif split == 'test':
        iwslt_dataset = datasets.load_dataset("iwslt2017", f"iwslt2017-en-fr")['validation']
        src = [sample['translation']['en'] for sample in iwslt_dataset]
        ref = [sample['translation']['fr'] for sample in iwslt_dataset]
    for s, r in zip(src, ref):
        prompt = f'Translate from English to French.\n{s}'
        data[prompt]['responses'] = [r]
        data[prompt]['pairs'] = [(0, 0)]
        data[prompt]['sft_target'] = r
    return data


def get_se(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data

def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def get_dataset(name: str, split: str, dataset_mode: str = 'single', num_hyps: int = 8, with18: bool = False, silent: bool = False, cache_dir: str = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif name == 'wmt':
        data = get_wmt(split, mode=dataset_mode, num_hyps=num_hyps, with18=with18)
    elif name == 'flores':
        data = get_flores(split, silent=silent, cache_dir=cache_dir)
    elif name =='wmt_sft':
        data = get_wmt_sft(split, silent=silent, cache_dir=cache_dir)
    elif name == 'iwslt_sft':
        data = get_iwslt_sft(split, silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       dataset_mode: str = 'single',
                       num_hyps: int = 8,
                       with18: bool = False,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       shuffle_pair: bool = False,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, dataset_mode, num_hyps=num_hyps, with18=with18, silent=silent, cache_dir=cache_dir).items():
                if shuffle_pair:
                    for pair in data['pairs']:
                        flat_data.append((prompt, data['responses'], [pair], data['sft_target'], truncation_mode))
                else:
                    flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break

                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True