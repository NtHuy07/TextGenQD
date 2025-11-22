import json
from typing import Optional

import numpy as np
import random
import os
from itertools import product

from openelm.configs import QDEnvConfig, QDConfig
from openelm.environments.base import BaseEnvironment
from openelm.environments.prompt.prompt import PromptGenotype
from openelm.mutation_model import MutationModel
from openelm.evaluation_metrics import evaluate_metrics

def get_str_between(sub1, sub2, test_str):
    idx1 = test_str.index(sub1)
    idx2 = test_str.index(sub2)

    res = ''
    # getting elements in between
    for idx in range(idx1, idx2 + len(sub2)):
        res = res + test_str[idx]
    return res

def fix(json_str):
    idx = [pos for pos, char in enumerate(json_str) if char == '"']
    idx = idx[3:-1]
    for i in idx:
        json_str[idx] = "'"
    return json_str

def delete_char(str, char):
    if char in str:
        str = str.replace(char, "")
    return str

class CommonGenGenotype(PromptGenotype):

    def __init__(
        self, commongen: str, length: int = None, formal: float = None, maturity = None,
    ):
        self.commongen = commongen
        self.length = length
        self.formality = formal
        self.maturity = maturity

    def __str__(self) -> str:
        return self.commongen

    def evaluate(self, model, quality_metric, diversity_metric, ground_truth, original_text):
        if quality_metric == "ai-score":
            quality = None
            while quality is None:
                try:
                    quality_prompts = f"Text:{self.commongen}\n On a scale from 0 to 10, rate how well (accurate and sound good) the text covers all the target words: {original_text}. (0 = worst, 10 = best). Respond only in valid JSON with the key 'quality'. Do not include any explanation or additional text."
                    quality = model.gen(quality_prompts)
                    self.quality = round(float(json.loads(get_str_between("{", "}", quality))["quality"])/10, 4)
                except:
                    print("Retry scoring")
        elif quality_metric == "logit-score":
            quality_prompts = [
                f"Text:{self.commongen}\n Does this text contain all the words: {original_text} and sound good? Answer yes or no.\n",
                ]
            self.quality = round(float(model.gen(quality_prompts).detach().cpu().numpy()), 4)
            
        predictions, references = [self.commongen], [ground_truth]
        self.eval_metrics = evaluate_metrics(predictions=predictions, references=references)

        
        if diversity_metric == "ai-score":
            formality_str = f"""{self.commongen}\n
            Rate the formality level of the above paragraph on a scale from 0 to 10 (0 = least formal, 10 = most formal). Respond only in valid JSON with the key "formality". Do not include any explanation or additional text.
            """

            maturity_str = f"""{self.commongen}\n
            Rate the maturity level of the intended audience for the tone of this paragraph on a scale from 0 to 100 (0 = child, 100 = adult). Respond only in valid JSON with the key "maturity". Do not include any explanation or additional text.
            """
            formality_result = model(formality_str)
            maturity_result = model(maturity_str)
            try:
                self.length = round(float(len(self.commongen.split())), 4)
                self.formality = round(float(json.loads(get_str_between("{", "}", formality_result))["formality"]), 4)
                self.age = round(float(json.loads(get_str_between("{", "}", maturity_result))["maturity"]), 4)

                return float(self.quality)
            except:
                return -np.inf
            
        elif diversity_metric == "logit-score":
            opinions_pairs = [['formal', 'informal'], ['adult', 'child']]
            prob_lists = []
            for k, opinions_pair in enumerate(opinions_pairs):
                if k == 0:
                    prompt_list = [f"{self.commongen}\nWhat kind of language is this paragraph closest to from the following list: {opinions_pair}?\nAnswer:"]
                elif k == 1:
                    prompt_list = [f"{self.commongen}\nWhat audiences is this paragraph more suitable for from the following list: {opinions_pair}?\nAnswer:"]
                opinions_prob = model.get_opinion_score(prompt_list, opinions_pair)
                opinions_prob = opinions_prob[0]/opinions_prob.sum()
                prob_lists.append(opinions_prob.detach().cpu().numpy())
            
            self.length = round(float(len(self.commongen.split())), 4)
            self.formality = round(prob_lists[0] * (10 - 0), 4)
            self.maturity = round(prob_lists[1] * (100 - 0), 4)

        return float(self.quality)

    def to_phenotype(self) -> Optional[np.ndarray]:
        if isinstance(self.length, float) and isinstance(self.formality, float) and isinstance(self.maturity, float):
            return np.array([float(self.length), float(self.formality), float(self.maturity)])
        return None
    

class CommonGenEvolution(BaseEnvironment[CommonGenGenotype]):

    def __init__(
        self,
        config: QDEnvConfig,
        qd_config: QDConfig,
        model_config,
        mutation_model: MutationModel,
    ):
        self.config: QDEnvConfig = config
        self.qd_config: QDConfig = qd_config
        self.model_config = model_config
        self.batch_size = self.config.batch_size
        self.genotype_space = np.array(self.config.behavior_space).T
        self.genotype_ndim = self.genotype_space.shape[1]
        self.mutation_model = mutation_model
        self.eval_model = self.mutation_model
        del mutation_model

        self.quality_metric = self.config.quality_metric
        self.diversity_metric = self.config.diversity_metric            

        self.tokenizer = self.mutation_model.tokenizer

        self.rng = np.random.default_rng(self.config.seed)

        self.count = 0

    def set_data(self, data):
        self.original_text = data['concepts']
        self.ground_truth = data['target']

    def quality_metric_fn(self, predictions=None, references=None):
        return self.metric_fn.compute(predictions=predictions, references=references, use_stemmer=True)[f'{self.quality_metric}']
        
    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        self.rng = rng_state

    def construct_prompt(self, commongen: Optional[CommonGenGenotype] = None, nonzero=None) -> dict[str, str]:
        formality_choices = ['informal', 'casual', 'formal']
        maturity_choices = ['child', 'young people', 'adult']

        if nonzero is not None and nonzero.sum() != int(self.qd_config.map_grid_size[0]**len(self.config.behavior)):
            nonzero = np.invert(nonzero).flatten().astype(int)
            idx_arr = np.arange(nonzero.shape[0])
            prob = nonzero/nonzero.sum()
            idx_choice = np.random.choice(idx_arr, p=prob)

            bs = [idx_choice // self.qd_config.map_grid_size[0], idx_choice % self.qd_config.map_grid_size[0]]

            pivot_b = 100 / self.qd_config.map_grid_size[0]

            i = 0
            if 0 in self.config.behavior:
                max_length = self.config.behavior_space[0][1]
                pivot = max_length / self.qd_config.map_grid_size[0]
                target_length = np.random.randint(1+int(pivot*bs[i]),int(pivot*(bs[i]+1))), np.random.randint(1+int(pivot_b*bs[i]),int(pivot_b*(bs[i]+1)))
                length_str = f'in exactly {target_length} words '
                i += 1
            else:
                target_length = None
                length_str = ''
            if 1 in self.config.behavior:
                real_b = np.random.randint(1+int(pivot_b*bs[i]),int(pivot_b*(bs[i]+1)))
                target_formality = formality_choices[int(real_b/(100/len(formality_choices)))]
                formality_str = f'with a {target_formality} tone '
                i += 1
            else:
                target_formality = None
                formality_str = ''
            if 2 in self.config.behavior:
                real_b = np.random.randint(1+int(pivot_b*bs[i]),int(pivot_b*(bs[i]+1)))
                target_maturity = maturity_choices[int(real_b/(100/len(maturity_choices)))]
                maturity_str = f'for a {target_maturity} '
            else:
                target_maturity = None
                maturity_str = ''

        else:
            i = 0
            if 0 in self.config.behavior:
                target_length = np.random.randint(1,self.config.behavior_space[0][1])
                length_str = f'in exactly {target_length} words '
                i += 1
            else:
                target_length = None
                length_str = ''
            if 1 in self.config.behavior:
                target_formality = self.rng.choice(formality_choices)
                formality_str = f'with a {target_formality} tone '
                i += 1
            else:
                target_formality = None
                formality_str = ''
            if 2 in self.config.behavior:
                target_maturity = self.rng.choice(maturity_choices)
                maturity_str = f'for a {target_maturity} '
            else:
                target_maturity = None
                maturity_str = ''
                
        if self.model_config.system_prompt:
            system_prompt_str = "You are a chatbot who generate text from provided words. Only write the generated text."
            addi_prompt_str = "Only write the generated text."
        else:
            system_prompt_str = None
            addi_prompt_str = "Only write the generated text."
        
        if commongen is None:
            instruction_str = f"Write a paragraph containing all the provided words " + maturity_str + formality_str + length_str + addi_prompt_str
            prompt_str = f"Provided words:\n {self.original_text}\n{instruction_str}"
        else:
            prob = random.random()
            if prob <= self.qd_config.diversity_mutation_prob: # Diversity mutation without original context
                instruction_str = f"Translate this text into another text " + maturity_str + formality_str + length_str + f"but still contains these words: {self.original_text}." + addi_prompt_str
                prompt_str = f"{commongen.commongen}.\n\n{instruction_str}"

            else: # Quality mutation with original context
                instruction_str = f"Rewrite this text into another text that sound more natural and contains these words: {self.original_text} without changing the tone and length." + addi_prompt_str
                prompt_str = f"Text: {commongen.commongen}.\n\n{instruction_str}"

        return {
            "prompt": prompt_str,
            "system_prompt": system_prompt_str,
            "target_length": target_length,
            "target_formality": target_formality,
            "target_maturity": target_maturity,
        }

    def random(self) -> list[CommonGenGenotype]:
        # Mutate seed, and pick random target genre and poem.
        prompt_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        results = []
        for prompt in prompt_list:
            result = None
            while result is None:
                try:
                    result = self.mutation_model.chatbot_gen(prompts=prompt['prompt'], system_prompts=prompt['system_prompt'])
                    result = result.split('<|eot_id|>')[0]
                    if self.config.debug:
                        print("\nGenerate new gene successfully")
                except:
                    if self.config.debug:
                        print("\nFalied to generate new gene. Retrying...")
                    pass
            results.append(result)
        return [CommonGenGenotype(commongen=c) for c in results]

    def mutate(self, genomes: list[CommonGenGenotype], nonzero=None) -> list[CommonGenGenotype]:
#         prompt_list: list[dict[str, str]] = list(map(self.construct_prompt, genomes, nonzero))
        prompt_list: list[dict[str, str]] = [self.construct_prompt(genome, nonzero) for genome in genomes]
        results = []
        for prompt in prompt_list:
            result = None
            while result is None:
                try:
                    result = self.mutation_model.chatbot_gen(prompts=prompt['prompt'], system_prompts=prompt['system_prompt'])
                    result = result.split('<|eot_id|>')[0]
                    if self.config.debug:
                        print("\nGenerate new gene successfully")
                except:
                    if self.config.debug:
                        print("\nFalied to generate new gene. Retrying...")
                    pass
            results.append(result)
        return [CommonGenGenotype(commongen=c) for c in results]

    def fitness(self, x: CommonGenGenotype) -> float:

        self.count += 1

        evaluation = x.evaluate(self.eval_model, diversity_metric=self.diversity_metric, 
                                ground_truth=self.ground_truth, original_text=self.original_text)
        if self.config.debug:
            print(f"-- Gene --\n{x.commongen}\n-- Fitness: {evaluation} --\n-- Behavior: {x.to_phenotype()} --")
            print(f"Eval Metrics: {x.eval_metrics}")

        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(self.config.output_dir + "/all_genomes_log.txt", "+a") as log_file:
            log_file.write(f"\n\n[[ID: {self.count}]]\n-- Gene --\n{x.commongen}\n-- Fitness: {evaluation} --\n-- Behavior: {x.to_phenotype()} --")
            log_file.write(f"\n-- Eval Metrics: {x.eval_metrics} --")
        
        return evaluation