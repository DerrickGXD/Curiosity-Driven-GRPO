from typing import List, Callable, Union
import os
import random
import numpy as np
from multiprocessing import Pool

import nltk
import sacrebleu

class SelfBleuReward(object):

    def __init__(self, 
                 grams: List[int] = [3, 4, 6], 
                 sample_size: int = -1,
                 tokenizer: Callable = nltk.word_tokenize, is_reasoning_pattern : bool = False) -> None:
        print("BLEU sample size: ", sample_size)
        self.references = {} # query prompt : reference
        self.grams = grams
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.is_reasoning_pattern = is_reasoning_pattern


    def reasoning_pattern_append_reference(self, prompt, ref, reasoning_pattern):
        if(prompt not in self.references):
            self.references[prompt] = {}

        if(reasoning_pattern not in self.references[prompt]):
            self.references[prompt][reasoning_pattern] = []
        
        self.references[prompt][reasoning_pattern].append([ref])
 

    def non_reasoning_pattern_append_reference(self, prompt, ref):
        if(prompt not in self.references):
            self.references[prompt] = []
        
        self.references[prompt].append([ref])


    def append_reference(self, prompt, ref, reasoning_pattern=None):
        if(self.is_reasoning_pattern):
            return self.reasoning_pattern_append_reference(prompt, hypotheses, reasoning_pattern)
        else:
            return self.non_reasoning_pattern_append_reference(prompt, hypotheses)


    def reasoning_pattern_call(self, prompt, hypotheses, reasoning_pattern):
        if self.sample_size > 0:
            sample_size = min(len(self.references[prompt][hypotheses]), self.sample_size)
            references = random.sample(self.references[prompt][hypotheses], k=sample_size)
        else:
            if(prompt in self.references):
                if(reasoning_pattern in self.references[prompt]):
                    references = self.references[prompt][reasoning_pattern]
                else:
                    references = []
            else:
                references = []

        if(len(references)>0):
            max_score = 0.0
            for ref in references:
                cur_score = sacrebleu.corpus_bleu(hypotheses, [ref]).score / 100
                if(cur_score > max_score):
                    max_score = cur_score
            score = max_score
        else:
            score = 0.0

        return score


    def non_reasoning_pattern_call(self, prompt, hypotheses):
        if self.sample_size > 0:
            sample_size = min(len(self.references[prompt]), self.sample_size)
            references = random.sample(self.references[prompt], k=sample_size)
        else:
            if(prompt in self.references):
                references = self.references[prompt]
            else:
                references = []

        if(len(references)>0):
            max_score = 0.0
            for ref in references:
                cur_score = sacrebleu.corpus_bleu(hypotheses, [ref]).score / 100
                if(cur_score > max_score):
                    max_score = cur_score
            score = max_score
        else:
            score = 0.0


        return score



    def __call__(self, prompt, hypotheses, reasoning_pattern=None):
        # weights = {f"{n}-gram": ([1. / n] * n) for n in self.grams}

        if(self.is_reasoning_pattern):
            return self.reasoning_pattern_call(prompt, hypotheses, reasoning_pattern)
        else:
            return self.non_reasoning_pattern_call(prompt, hypotheses)