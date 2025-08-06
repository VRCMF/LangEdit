from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import time
import torch
import numpy as np
import ipdb

MAX_NUMBER_OF_FEW_SHOTS = 100
## IMPORTANT, few shot learning is important as it allow the answer coming out from the model to be formatted. 

class XNLIEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation', lang_s='en'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {MAX_NUMBER_OF_FEW_SHOTS}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/xnli_{}.pkl'.format(lang_s), number_of_few_shots, number_of_tests, 'xnli') 
        self._initialize_prompts(lang_s)
        
    # def _initialize_prompts(self):
    #     self.postfix_prompt = 'True or False? answer:' 
    #     self.few_shot_context = ""
    #     for _, few_shot in enumerate(self.few_shots):
    #         self.few_shot_context += f'{few_shot["sentence1"]} entails the {few_shot["sentence2"]} {self.postfix_prompt} {("True" if few_shot["label"] == "entailment" else "False")}\n' 

    def _initialize_prompts(self, lang_s):
        # 可以尝试加上 “ sent1 " entails the " sent2
        self.few_shot_context = []
        if lang_s == 'en':
            self.postfix_prompt = 'Yes or No? Answer:' 
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f'{few_shot["sentence1"]} entails the {few_shot["sentence2"]} {self.postfix_prompt} {("Yes" if few_shot["label"] == "entailment" else "No")}\n')  
        elif lang_s == 'fr':
            self.postfix_prompt = 'Oui ou Non? Réponse:' 
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f'{few_shot["sentence1"]} implique que {few_shot["sentence2"]} {self.postfix_prompt} {("Oui" if few_shot["label"] == "entailment" else "Non")}\n')  
        elif lang_s == 'nl':
            self.postfix_prompt = 'Ja of Nee? Antwoord:' 
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f'{few_shot["sentence1"]} houdt in dat {few_shot["sentence2"]} {self.postfix_prompt} {("Ja" if few_shot["label"] == "entailment" else "Nee")}\n')  
        elif lang_s == 'es':
            self.postfix_prompt = 'Sí o No? Respuesta:' 
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f'{few_shot["sentence1"]} implica que {few_shot["sentence2"]} {self.postfix_prompt} {("Sí" if few_shot["label"] == "entailment" else "No")}\n')  
        elif lang_s == 'de':
            self.postfix_prompt = 'Ja oder Nein? Antwort:' 
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f'{few_shot["sentence1"]} impliziert {few_shot["sentence2"]} {self.postfix_prompt} {("Ja" if few_shot["label"] == "entailment" else "Nein")}\n')  
        elif lang_s == 'zh':
            self.postfix_prompt = '是 或者 否？回答:' 
            for _, few_shot in enumerate(self.few_shots):
                self.few_shot_context.append(f'{few_shot["sentence1"]} 蕴含着 {few_shot["sentence2"]} {self.postfix_prompt} {("是" if few_shot["label"] == "entailment" else "否")}\n')  

    def _create_prompt(self, example, gen_len, lang_s):
        if lang_s == 'en':
            question = f'{example["sentence1"]} entails the {example["sentence2"]} " {self.postfix_prompt}'
        elif lang_s == 'fr':
            question = f'{example["sentence1"]} implique que {example["sentence2"]} " {self.postfix_prompt}'
        elif lang_s == 'nl':
            question = f'{example["sentence1"]} houdt in dat {example["sentence2"]} " {self.postfix_prompt}'
        elif lang_s == 'es':
            question = f'{example["sentence1"]} implica que {example["sentence2"]} " {self.postfix_prompt}'
        elif lang_s == 'de':
            question = f'{example["sentence1"]} impliziert {example["sentence2"]} " {self.postfix_prompt}'
        elif lang_s == 'zh':
            question = f'{example["sentence1"]} 蕴含着 {example["sentence2"]} " {self.postfix_prompt}'
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
        return input_prompt, example['sentence1'], example['sentence2'], self._get_label(example['label'])
    
    # def _get_answer(self, generated_text, lang_s):
    #     answer_text = generated_text.split(self.postfix_prompt)[-1].strip().strip()

    #     if 'true' in answer_text.lower():
    #         return 1
    #     elif 'false' in answer_text.lower():
    #         return 0
    #     return -1

    def _get_answer(self, generated_text, lang_s):
        answer_text = generated_text.split(self.postfix_prompt)[-1].strip().strip()
        
        if lang_s == 'en':
            if 'yes' in answer_text.lower():
                return 1
            elif 'no' in answer_text.lower():
                return 0
        elif lang_s == 'fr':
            if 'oui' in answer_text.lower():
                return 1
            elif 'non' in answer_text.lower():
                return 0
        elif lang_s == 'es':
            if 'sí' in answer_text.lower():
                return 1
            elif 'no' in answer_text.lower():
                return 0
        elif lang_s == 'de':
            if 'ja' in answer_text.lower():
                return 1
            elif 'nein' in answer_text.lower():
                return 0
        elif lang_s == 'nl':
            if 'ja' in answer_text.lower():
                return 1
            elif 'nee' in answer_text.lower():
                return 0
        elif lang_s == 'zh':
            if '是' in answer_text.lower():
                return 1
            elif '否' in answer_text.lower():
                return 0
        return -1

    def _get_label(self, example_label):
        if 'entailment' == example_label:
            return 1
        return 0

    def evaluate(self, gen_len = 3, print_logs = False, lang_s='en'):
        
        if lang_s == 'en':
            true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Yes', 'No'])
        elif lang_s == 'fr':
            true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Oui', 'Non'])
        elif lang_s == 'es':
            true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Sí', 'No'])
        elif lang_s == 'de':
            true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Ja', 'Nein'])
        elif lang_s == 'nl':
            true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Ja', 'Nee'])
        elif lang_s == 'zh':
            true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['是', '否'])

        # true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['True', 'False'])

        if 'llama' in self.model.config._name_or_path.lower():
            true_tok = true_tok[1:]
            false_tok = false_tok[1:]

        true_len, false_len = (len(n) for n in [true_tok, false_tok])
        # suffixes = {0: ['True', true_tok, true_len], 1: ['False', false_tok, false_len]}

        if lang_s == 'en':
            suffixes = {0: ['Yes', true_tok, true_len], 1: ['No', false_tok, false_len]}
        elif lang_s == 'fr':
            suffixes = {0: ['Oui', true_tok, true_len], 1: ['Non', false_tok, false_len]}
        elif lang_s == 'es':
            suffixes = {0: ['Sí', true_tok, true_len], 1: ['No', false_tok, false_len]}
        elif lang_s == 'de':
            suffixes = {0: ['Ja', true_tok, true_len], 1: ['Nein', false_tok, false_len]}
        elif lang_s == 'nl':
            suffixes = {0: ['Ja', true_tok, true_len], 1: ['Nee', false_tok, false_len]}
        elif lang_s == 'zh':
            suffixes = {0: ['是', true_tok, true_len], 1: ['否', false_tok, false_len]}

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
            input_prompt, sentence1, sentence2, label = self._create_prompt(example, gen_len, lang_s)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)
            # if lang_s == 'de':
            #     ipdb.set_trace()
            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])

            if 'llama' in self.model.config._name_or_path.lower():
                prefix_tok_len = prefix_tok_len - 1

            max_len = input_prompt_ids.shape[1] + gen_len
            output = self.model.generate(input_prompt_ids,max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self._get_answer(generated_text, lang_s)

            predictions.append(answer)
            labels.append(label)

            #### EVALUATE NEW ACC 
            
            probs = [0 for _ in suffixes.keys()]
            gen_texts = [0 for _ in suffixes.keys()]

            for i in range(len(suffixes.keys())):
                print(suffixes[i][0])
                prompt_tok = self.tokenizer([f"{input_prompt} {suffixes[i][0]}"], return_tensors="pt").to('cuda')

                with torch.no_grad():
                    logits = self.model(**prompt_tok).logits

                if 'llama' in self.model.config._name_or_path.lower():
                    logits = logits[:, 1:, :]

                cur_len = suffixes[i][2]

                for j in range(cur_len):
                    cur_tok = suffixes[i][1][j]
                    probs[i] += -torch.nn.functional.log_softmax(
                    logits[0, prefix_tok_len + j - 1, :], dim=0
                    )[cur_tok].item()
                probs[i] /= cur_len
                gen_texts[i] = self.tokenizer.decode(logits[0, prefix_tok_len - 1 : prefix_tok_len + cur_len - 1, :].argmax(dim = -1))

            prob_true = np.exp(-probs[0])
            prob_false = np.exp(-probs[1])

            print(f"prob_true: {prob_true}, prob_false: {prob_false}")

            answer_new = 1 if prob_true > prob_false else 0
            predictions_new.append(answer_new)
            print(f"prediction: {answer}, true: {label}")
            if answer == -1:
                invalid += 1
            else:

                if answer == label:
                    correct += 1

                    if label == 1:
                        pos_correct += 1
                    elif label == 0:
                        neg_correct += 1

                else:
                    incorrect += 1

                    if label == 1:
                        pos_incorrect += 1
                    elif label == 0:
                        neg_incorrect += 1

            exp_temp_dict = {
                'sentence1': sentence1,
                'sentence2': sentence2,
                'input_prompt': input_prompt_text,
                'true_answer': 'True' if label == 1 else 'False', 
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': answer == label,
                'prob_true': prob_true,
                'prob_false': prob_false,
                'highest_probability_answer': 'True' if answer_new == 1 else 'False', 
                'correct_new': answer_new == label,
            }
            stored_generations.append(exp_temp_dict)

            if print_logs:
                mcc = matthews_corrcoef(labels, predictions)
                f1 = f1_score(labels, predictions, average='weighted')
                print(generated_text)
                print(correct, incorrect, invalid, s+1, '|', pos_correct, neg_correct, '|', pos_incorrect, neg_incorrect, '|ACC: ', correct / (correct + incorrect + invalid), '|MCC:', mcc, '|F1:', f1)
                print('--'*50)


        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        f1_new = f1_score(labels, predictions_new, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s+1,
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end-start,
        }

        return result_dict, stored_generations

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    for lang_s in ['en', 'de', 'nl', 'es', 'fr', 'zh']:
        xnli_eval = XNLIEval(model, tokenizer, lang_s=lang_s, number_of_tests=10, number_of_few_shots=2)
        xnli_eval.evaluate(print_logs='True', lang_s=lang_s)
        # ipdb.set_trace()