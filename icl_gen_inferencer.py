"""Direct Generation Inferencer"""

import json
import torch
import evaluate
#from bleurt import score

from openicl import PromptTemplate
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from openicl.utils.api_service import *
from openicl.utils.icl_common_utils import get_dataloader, get_generation_prompt_list_from_retriever_indices
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from tqdm import tqdm
from transformers import PretrainedConfig
from accelerate import Accelerator

logger = get_logger(__name__)


class GenInferencer(BaseInferencer):
    """Generation In-context Learning Inferencer Class
        In-context Learning Inferencer for Directly Generation.
        
    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class. 
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM. 
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file. 
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file. 
        api_name (:obj:`str`, optional): Name of API service. 
        call_api (:obj:`bool`): If ``True``, an API for LM models will be used, determined by :obj:`api_name`.   
        gen_field_replace_token (:obj:`str`, optional): Used to replace the generation field token when generating prompts.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the :obj:`model.generate()` method. 
    """

    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 model_ckpt: Optional[str] = None,
                 num_beam = 1,
                 len_penalty = 1.0,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 gen_field_replace_token: Optional[str] = '',
                 generation_kwargs={"max_new_tokens": 100},
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,
                 num_hyps = 8,
                 temp = 0.9,
                 calc_prob = False,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, model_ckpt, num_beam, len_penalty, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs
        self.num_hyps = num_hyps
        self.temp = temp
        self.calc_prob = calc_prob

    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None, force_words=None) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = GenInferencerOutputHandler(num, self.accelerator)
        index = 0

        sacrebleu = evaluate.load("sacrebleu")
        #scorer = score.BleurtScorer('BLEURT-20')
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input 
        prompt_list = get_generation_prompt_list_from_retriever_indices(ice_idx_list, retriever, self.tokenizer,
                                                                        self.gen_field_replace_token,
                                                                        max_model_token_num=None,
                                                                        ice_template=ice_template,
                                                                        prompt_template=prompt_template)
        output_handler.save_orgin_prompts(prompt_list)

        # 4. Wrap prompts with Dataloader
        dataloader = get_dataloader(prompt_list, self.batch_size)
        
        # 5. Inference for prompts in each batch 
        logger.info("Starting inference process...")
        for entry in tqdm(dataloader, disable=not self.is_main_process):
            # 5-1. Inference with local model
            if not self.call_api:
                with torch.no_grad():
                    tokenized_data = self.tokenizer.batch_encode_plus(entry, padding=True, return_tensors='pt').to(
                        self.device)
                    prompt_len = int(tokenized_data.attention_mask.shape[1])
                    if 't5' in self.model_name or 'mt0' in self.model_name:
                        prompt_len = 0
                    if force_words is not None:
                        force_words_ids = [
                            self.tokenizer(force_words).input_ids,
                        ]
                        outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      force_words_ids=force_words_ids,
                                                      num_beams=10,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                    elif self.num_beam > 0:
                        outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      num_beams=self.num_beam,
                                                      length_penalty=self.len_penalty,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                    else:
                        if self.calc_prob:
                            outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                          do_sample=True,
                                                          temperature=self.temp,
                                                          epsilon_cutoff=0.02,
                                                          num_return_sequences=self.num_hyps,
                                                          output_scores=True,
                                                          return_dict_in_generate=True,
                                                          attention_mask=tokenized_data.attention_mask,
                                                          eos_token_id=self.tokenizer.eos_token_id,
                                                          pad_token_id=self.tokenizer.pad_token_id,
                                                          **self.generation_kwargs)
                            logits = outputs.scores
                            probabilities = [torch.nn.functional.softmax(logit, dim=-1) for logit in logits]
                            outputs = outputs.sequences
                        else:
                            outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      do_sample=True,
                                                      temperature=self.temp,
                                                      num_return_sequences=self.num_hyps,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                    outputs = outputs.tolist()

                    complete_output = self.tokenizer.batch_decode(outputs[:], skip_special_tokens=True)
                    generated = self.tokenizer.batch_decode([output[prompt_len:] for output in outputs],
                                                            skip_special_tokens=True)
                    
                    scores = []
                    if self.calc_prob:
                        for i, output in enumerate(outputs):
                            selected_probs = []
                            for prob, gen_id in zip(probabilities, output[prompt_len:]):
                                if gen_id != self.tokenizer.pad_token_id:
                                    selected_probs.append(prob[i][gen_id])
                            log_probs = torch.log(torch.tensor(selected_probs))
                            sentence_log_prob = log_probs.sum(-1)
                            # print(sentence_log_prob.item())
                            scores.append(torch.exp(sentence_log_prob).item())
                    
                    


                    #generated = [hyp for hyp in generated if hyp.strip() != '']
                    #scores = [0.] * len(generated)
                    #for i, hyp in enumerate(generated):
                    #    for ref in generated:
                    #        scores[i] += sacrebleu.compute(predictions=[hyp], references=[[ref]])['score']
                    #for i, hyp in enumerate(generated):
                    #    scores[i] = sum(scorer.score(references=generated, candidates=[hyp] * len(generated)))
                    #max_idx = scores.index(max(scores))
                    #generated = [generated[max_idx]]
                    #complete_output = [complete_output[max_idx]]

            # 5-2. Inference with remote API
            else:
                complete_output, generated = api_get_tokens(self.api_name, entry)
            generated = [generated]
            scores = [scores]
            # 5-3. Save current output
            for prediction, output, prob in zip(generated, complete_output, scores):
                output_handler.save_prediction_and_output(prediction, output, prob, index)
                index = index + 1

        # 6. Output 
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)
        return [sample['prediction'] for sample in output_handler.results_dict.values()]
