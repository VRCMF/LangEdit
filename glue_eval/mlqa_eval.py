from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
# from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import time
import torch
import numpy as np
import ipdb
import jieba
from collections import Counter

MAX_NUMBER_OF_FEW_SHOTS = 100


import re
from collections import Counter

def normalize_text(text):
    """
    Lowercase, remove punctuation, articles, and extra whitespace from the text.
    """
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)  # remove articles
    text = re.sub(r'[^a-z0-9\s]', '', text)      # remove punctuation
    text = ' '.join(text.split())                # remove extra spaces
    return text

def exact_match(prediction, ground_truth):
    """
    Compute Exact Match (EM).
    """
    # ipdb.set_trace()
    # return int(normalize_text(prediction) == normalize_text(ground_truth))
    return int(prediction == ground_truth)

def f1_score_char(pred, ref):
    # pred = normalize_text(pred)
    # ref = normalize_text(ref)
    
    pred_tokens = list(pred)
    ref_tokens = list(ref)
    # ipdb.set_trace()
    common = Counter(pred_tokens) & Counter(ref_tokens)  # 统计共同 token
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def f1_score(prediction, ground_truth):
    """
    Compute F1 score based on token overlap.
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def f1_score_token(pred, ref):
    pred_tokens = jieba.lcut(pred)  # 对预测答案分词
    ref_tokens = jieba.lcut(ref)    # 对参考答案分词
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def qa_evaluate(predictions, ground_truths, lang_s):
    """
    Evaluate the model output using EM and Mean Token F1.
    """
    total_em = 0
    total_f1 = 0
    n = len(predictions)
    
    for pred, gt in zip(predictions, ground_truths):
        total_em += exact_match(pred, gt)
        if lang_s == 'zh':
            total_f1 += f1_score_token(pred, gt)
        else:
            total_f1 += f1_score(pred, gt)
    
    em_score = total_em / n
    mean_f1 = total_f1 / n
    return em_score, mean_f1

# Example usage
# predictions = [
#     "The capital of France is Paris.",
#     "Mount Everest is the tallest mountain.",
#     "The Amazon River is the longest in the world."
# ]

# ground_truths = [
#     "Paris is the capital of France.",
#     "Mount Everest is the tallest mountain in the world.",
#     "The Amazon River is the longest river in the world."
# ]

# em, mean_f1 = evaluate(predictions, ground_truths)
# print(f"Exact Match (EM): {em:.2f}")
# print(f"Mean Token F1 Score: {mean_f1:.2f}")



class MLQAXEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation', lang_s='en'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/mlqa_{}.pkl'.format(lang_s),  number_of_few_shots, number_of_tests, 'mlqa') 
        self._initialize_prompts(lang_s)


    def _initialize_prompts(self, lang_s):
        if lang_s == 'en':
            self.prefix_prompt = 'Answer the following questions based on the provided context.\n'
            self.postfix_prompt = 'Answer:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}Context : {few_shot['context']}\nQuestion: {few_shot['question']}\nAnswer: {few_shot['answer']}\n")
        elif lang_s == 'zh':
            self.prefix_prompt = '根据提供的上下文回答以下问题。\n'
            self.postfix_prompt = '回答:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}语境: {few_shot['context']}\n问题: {few_shot['question']}\n回答: {few_shot['answer']}\n")
        elif lang_s == 'es':
            self.prefix_prompt = 'Responda las siguientes preguntas basándose en el contexto proporcionado.\n'
            self.postfix_prompt = 'Respuesta:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}Contexto: {few_shot['context']}\nPregunta: {few_shot['question']}\nRespuesta: {few_shot['answer']}\n")
        elif lang_s == 'de':
            self.prefix_prompt = 'Beantworten Sie die folgenden Fragen basierend auf dem bereitgestellten Kontext.\n'
            self.postfix_prompt = 'Antwort:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}Kontext: {few_shot['context']}\nFrage: {few_shot['question']}\nAntwort: {few_shot['answer']}\n")

    # def _create_prompt(self, example):
    #     prompt = 'Sentence 1: ' + example['sentence1'] + '\n'
    #     prompt += 'Sentence 2: ' + example['sentence2'] + '\n'

    #     input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

    #     return input_prompt, example['sentence1'], example['sentence2'], example['label']
    
    def _create_prompt(self, example, gen_len, lang_s):
        if lang_s == 'en':
            prompt = 'Context: ' + example['context'] + '\n'
            prompt += 'Question: ' + example['question'] + '\n'
        elif lang_s == 'zh':
            prompt = '语境: ' + example['context'] + '\n'
            prompt += '问题: ' + example['question'] + '\n'
        elif lang_s == 'es':
            prompt = 'Contexto: ' + example['context'] + '\n'
            prompt += 'Pregunta: ' + example['question'] + '\n'
        elif lang_s == 'de':
            prompt = 'Kontext: ' + example['context'] + '\n'
            prompt += 'Frage: ' + example['question'] + '\n'
        question = self.prefix_prompt + prompt + self.postfix_prompt
        question_token_length = len(self.tokenizer(question)["input_ids"])
        # ipdb.set_trace()
        remaining_token_length = MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP[self.model.config._name_or_path.lower().split('/')[-1]] - question_token_length - gen_len
        actual_few_shot = ""
        for few_shot in self.few_shot_context:
            few_shot_token_length = len(self.tokenizer(few_shot)["input_ids"])
            remaining_token_length -= few_shot_token_length
            if remaining_token_length < 0:
                break 
            actual_few_shot += few_shot
        input_prompt = actual_few_shot + question
        return input_prompt, example['context'], example['question'], example['answer']


    def evaluate(self, gen_len = 20, print_logs = False, lang_s='en'):

        # if lang_s == 'en':
        #     yes_tok, no_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Yes', 'No'])
        # elif lang_s == 'fr':
        #     yes_tok, no_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Oui', 'Non'])
        # elif lang_s == 'es':
        #     yes_tok, no_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Sí', 'No'])
        # elif lang_s == 'de':
        #     yes_tok, no_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Ja', 'Nein'])
        # elif lang_s == 'nl':
        #     yes_tok, no_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Ja', 'Nee'])

        # # yes_tok, no_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Yes', 'No'])

        # if 'llama' in self.model.config._name_or_path.lower():
        #     yes_tok = yes_tok[1:]
        #     no_tok = no_tok[1:]

        # yes_len, no_len = (len(n) for n in [yes_tok, no_tok])

        # suffixes = {0: ['Yes', yes_tok, yes_len], 1: ['No', no_tok, no_len]}

        # if lang_s == 'en':
        #     yes_len, no_len = (len(n) for n in [yes_tok, no_tok])
        #     suffixes = {0: ['Yes', yes_tok, yes_len], 1: ['No', no_tok, no_len]}
        # elif lang_s == 'fr':
        #     yes_len, no_len = (len(n) for n in [yes_tok, no_tok])
        #     suffixes = {0: ['Oui', yes_tok, yes_len], 1: ['Non', no_tok, no_len]}
        # elif lang_s == 'es':
        #     yes_len, no_len = (len(n) for n in [yes_tok, no_tok])
        #     suffixes = {0: ['Sí', yes_tok, yes_len], 1: ['No', no_tok, no_len]}
        # elif lang_s == 'de':
        #     yes_len, no_len = (len(n) for n in [yes_tok, no_tok])
        #     suffixes = {0: ['Ja', yes_tok, yes_len], 1: ['Nein', no_tok, no_len]}
        # elif lang_s == 'nl':
        #     yes_len, no_len = (len(n) for n in [yes_tok, no_tok])
        #     suffixes = {0: ['Ja', yes_tok, yes_len], 1: ['Nee', no_tok, no_len]}

        correct = 0
        incorrect = 0
        invalid = 0

        pos_correct = 0
        neg_correct = 0
        pos_incorrect = 0
        neg_incorrect = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []
        start = time.time()

        for s, example in enumerate(self.eval_dataset):

            input_prompt, context, question, label = self._create_prompt(example, gen_len, lang_s)
            # print(input_prompt)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])

            if 'llama' in self.model.config._name_or_path.lower():
                prefix_tok_len = prefix_tok_len - 1

            max_len = input_prompt_ids.shape[1] + gen_len
            output = self.model.generate(input_prompt_ids,max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # answer = self._get_answer(generated_text, lang_s)
            split_gen = generated_text.split(self.postfix_prompt)[-1].strip().strip().split('\n')[0]

            predictions.append(split_gen.lower())
            labels.append(label.lower())

            exp_temp_dict = {
                'context': context, 
                'question': question, 
                'input_prompt': input_prompt_text,
                'answer': 'Yes' if label == 1 else 'No',
                'generated_text': split_gen,
                'answer': label,
                }
            stored_generations.append(exp_temp_dict)

        em, mean_f1 = qa_evaluate(predictions, labels, lang_s)
        print("em : {} and f1 {}".format(em, mean_f1))
        result_dict = {
            # 'correct': correct,
            # 'incorrect': incorrect,
            # 'invalid': invalid,
            # 'total': s+1,
            'em_f1': em,
            'mean_token_f1': mean_f1,
            # 'f1_new': f1_new,
            # 'mcc': mcc,
            # 'time': end-start,
        }

        return result_dict, stored_generations

if __name__ == '__main__':
    # Load the tokenizer and model
    #model_name = 'EleutherAI/gpt-j-6b'
    #model_name = 'gpt2-xl'
    # model_name = '/data/akshat/lingua-models/Llama-2-7b-hf'
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    # for lang_s in ['nl']:
    str_ = ""
    for lang_s in ['en', 'de', 'es', 'zh']:
        # pawsx_eval = PAWSXEval(model, tokenizer, lang_s=lang_s, number_of_few_shots=2, number_of_tests=100)
        mlqa_eval = MLQAXEval(model, tokenizer, lang_s=lang_s, number_of_few_shots=2, number_of_tests=10)
        result_dict = mlqa_eval.evaluate(print_logs='True', lang_s=lang_s)
        str_ += str(result_dict[0]['em_f1']) + '#' + str(result_dict[0]['mean_token_f1']) + '\n'

    with open('yourpath', 'w', encoding='utf-8') as f:
        f.write(str_)

    
    