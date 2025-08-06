from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
# from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import time
import torch
import numpy as np
import ipdb
from sklearn.metrics import precision_score, recall_score, f1_score
import re
# from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


MAX_NUMBER_OF_FEW_SHOTS = 100

def convert_bio_to_entity(tag_seq):
    return [tag[2:] if tag.startswith(("B-", "I-")) else tag for tag in tag_seq]

def parse_output(output):
    predictions = {"PER": [], "LOC": [], "ORG": []}
    for label in predictions.keys():
        match = re.search(f"{label}: (.+)", output)
        if match:
            predictions[label].extend(match.group(1).split("@"))
    return predictions

classwise_size = 100

def flatten_labels(label_dict):
    flattened = []
    for label, items in label_dict.items():
        flattened.extend([(item, label) for item in items])
    return flattened

def assign_bio_labels(tokens, entities):
    bio_labels = ["O"] * len(tokens)
    for entity, label in entities:
        entity_tokens = entity.split()
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                bio_labels[i] = f"B-{label}"
                for j in range(1, len(entity_tokens)):
                    bio_labels[i+j] = f"I-{label}"
                break  # 一个实体只匹配一次
    return bio_labels

# true_labels = flatten_labels(ground_truth) flatten_labels(parse_output(tmp_example['label']))
# pred_labels = flatten_labels(predictions)

# 计算F1分数
def calculate_tp_fp_fn(true_labels, pred_labels):
    true_set = set(true_labels)
    pred_set = set(pred_labels)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    # precision = tp / (tp + fp) if tp + fp > 0 else 0
    # recall = tp / (tp + fn) if tp + fn > 0 else 0
    # f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return tp, fp, fn

class WIKIANNEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation', lang_s='en'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/wikiann_{}.pkl'.format(lang_s),  number_of_few_shots, number_of_tests, 'wikiann') 
        self._initialize_prompts(lang_s)


    def _initialize_prompts(self, lang_s):
        if lang_s == 'en':
            self.prefix_prompt = "Extract entities from the given text and entities to be recognized are: Person, Location, and Organization.\n"
            self.postfix_prompt = 'Answer:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}Text: {few_shot['text']}\nAnswer: {few_shot['label']}\n")
        elif lang_s == 'fr':
            self.prefix_prompt = "Extraire les entités du texte donné et les entités à reconnaître sont: Personne, Lieu, Organisation.\n"
            self.postfix_prompt = 'Répondre:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}Texte: {few_shot['text']}\Répondre: {few_shot['label']}\n")
        elif lang_s == 'es':
            self.prefix_prompt = "Extraiga entidades del texto dado y las entidades a reconocer son: Persona, Ubicación, Organización.\n"
            self.postfix_prompt = 'Respuesta:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}Texto: {few_shot['text']}\Respuesta: {few_shot['label']}\n")
        elif lang_s == 'de':
            self.prefix_prompt = "Entitäten aus dem gegebenen Text extrahieren. Zu erkennende Entitäten sind: Person, Ort, Organisation.\n"
            self.postfix_prompt = 'Antwort:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}Text: {few_shot['text']}\Antwort: {few_shot['label']}\n")
        elif lang_s == 'zh':
            self.prefix_prompt = "从给定的文本中提取实体，要识别的实体为：人, 地点 和 组织\n"
            self.postfix_prompt = '答案:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}文本: {few_shot['text']}\答案: {few_shot['label']}\n")
        elif lang_s == 'nl':
            self.prefix_prompt = "Extracteer entiteiten uit de gegeven tekst en de te herkennen entiteiten zijn: Persoon, Locatie, Organisatie.\n"
            self.postfix_prompt = 'Antwoord:'
            self.few_shot_context = []
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f"{self.prefix_prompt}Tekst: {few_shot['text']}\Antwoord: {few_shot['label']}\n")

    # def _create_prompt(self, example):
    #     prompt = 'Sentence 1: ' + example['sentence1'] + '\n'
    #     prompt += 'Sentence 2: ' + example['sentence2'] + '\n'

    #     input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

    #     return input_prompt, example['sentence1'], example['sentence2'], example['label']
    
    def _create_prompt(self, example, gen_len, lang_s):
        if lang_s == 'en':
            prompt = 'Text: ' + example['text'] + '\n'
        elif lang_s == 'fr':
            prompt = 'Texte: ' + example['text'] + '\n'
        elif lang_s == 'es':
            prompt = 'Texto: ' + example['text'] + '\n'
        elif lang_s == 'de':
            prompt = 'Text: ' + example['text'] + '\n'
        elif lang_s == 'zh':
            prompt = '文本: ' + example['text'] + '\n'
        elif lang_s == 'nl':
            prompt = 'Tekst: ' + example['text'] + '\n'
        question = self.prefix_prompt + prompt + self.postfix_prompt
        question_token_length = len(self.tokenizer(question)["input_ids"])
        remaining_token_length = MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP[self.model.config._name_or_path.lower().split('/')[-1]] - question_token_length - gen_len
        actual_few_shot = ""
        for few_shot in self.few_shot_context:
            few_shot_token_length = len(self.tokenizer(few_shot)["input_ids"])
            remaining_token_length -= few_shot_token_length
            if remaining_token_length < 0:
                break 
            actual_few_shot += few_shot
        input_prompt = actual_few_shot + question
        return input_prompt, example['text'], example['label']


    def evaluate(self, gen_len = 20, print_logs = False, lang_s='en'):

        
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

        pred_bio_labels_list = []
        gt_bio_labels_list = []

        tp = 0
        fp = 0
        fn = 0

        # ipdb.set_trace()

        for s, example in enumerate(self.eval_dataset):

            input_prompt, sentence1, label = self._create_prompt(example, gen_len, lang_s)
            # print(input_prompt)
            
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])

            if 'llama' in self.model.config._name_or_path.lower():
                prefix_tok_len = prefix_tok_len - 1

            max_len = input_prompt_ids.shape[1] + gen_len
            output = self.model.generate(input_prompt_ids,max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            split_gen = generated_text.split(self.postfix_prompt)[-1].strip().strip()
            answer = flatten_labels(parse_output(split_gen))
            gt = flatten_labels(parse_output(label))

            pred_bio_labels = assign_bio_labels(example['text'].split(), answer)
            gt_bio_labels = assign_bio_labels(example['text'].split(), gt)

            # pred_bio_labels_list.append(pred_bio_labels) 
            # gt_bio_labels_list.append(gt_bio_labels)
            pred_bio_labels_list += pred_bio_labels
            gt_bio_labels_list += gt_bio_labels

            predictions.append(answer)
            labels.append(gt)


            exp_temp_dict = {
                'text': input_prompt, 
                'input_prompt': input_prompt_text,
                'true_answer': str(label),
                # 'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': str(answer),
                # 'correct': answer == label,
                # 'prob_yes': prob_yes,
                # 'prob_no': prob_no,
                # 'highest_probability_answer': 'Yes' if answer_new == 1 else 'No', 
                # 'correct_new': answer_new == label,
                }
            stored_generations.append(exp_temp_dict)


        end = time.time()
        res_gt = convert_bio_to_entity(gt_bio_labels_list)
        res_pred = convert_bio_to_entity(pred_bio_labels_list)

        micro_f1 = f1_score(res_gt, res_pred, average='micro')
        macro_f1 = f1_score(res_gt, res_pred, average='macro')

        result_dict = {
            # 'correct': correct,
            # 'incorrect': incorrect,
            # 'invalid': invalid,
            # 'total': s+1,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            # 'f1_new': f1_new,
            # 'mcc': mcc,
            # 'time': end-start,
        }
        print("micro f1 : {} and macro f1 {}".format(micro_f1, macro_f1))
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
    for lang_s in ['en', 'de', 'nl', 'es', 'fr', 'zh']:
        # pawsx_eval = PAWSXEval(model, tokenizer, lang_s=lang_s, number_of_few_shots=2, number_of_tests=100)
        wikiann_eval = WIKIANNEval(model, tokenizer, lang_s=lang_s, number_of_few_shots=3, number_of_tests=10)
        result_dict = wikiann_eval.evaluate(print_logs='True', lang_s=lang_s)
        str_ += str(result_dict[0]['micro_f1']) + '#' + str(result_dict[0]['macro_f1']) + '\n'

    with open('/dodrio/scratch/projects/2024_069/mke/AlphaEdit/glue_eval/dataset/results/wikiann.txt', 'w', encoding='utf-8') as f:
        f.write(str_)
    
    