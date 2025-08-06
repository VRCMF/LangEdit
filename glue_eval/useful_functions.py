import pickle
import ipdb

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()
    

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

# def load_data_split(filename, split):
#     a_file = open(filename, "rb")
#     output = pickle.load(a_file)
#     a_file.close()
#     return output[:split], output[split:]

FEW_SHOT_TEST_SPLIT = 10

def load_data_split(filename, number_of_few_shots, number_of_tests, task):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    # ipdb.set_trace()
    assert number_of_few_shots <= FEW_SHOT_TEST_SPLIT, f"The largest number of few shot can only be 100, we received {number_of_few_shots}"
    if not number_of_tests is None:
        assert number_of_tests <= len(output) - FEW_SHOT_TEST_SPLIT,  f"The largest number of test for this task can only be {len(output) - FEW_SHOT_TEST_SPLIT}, we received {number_of_tests}"
    else:
        number_of_tests = len(output) - FEW_SHOT_TEST_SPLIT
    allow_few_shots, allow_tests = output[:FEW_SHOT_TEST_SPLIT], output[FEW_SHOT_TEST_SPLIT:]
    # ipdb.set_trace()
    if task == 'xnli':
        get_sublist = get_sublist_xnli
    elif task == 'pawsx':
        get_sublist = get_sublist_pawsx
    elif task == 'mlqa':
        get_sublist = get_sublist_mlqa
    elif task == 'wikiann':
        get_sublist = get_sublist_wikiann
    if number_of_few_shots > 0:
        return get_sublist(allow_few_shots, number_of_few_shots), allow_tests[:number_of_tests]
    else:
        return allow_few_shots[:number_of_few_shots], allow_tests[:number_of_tests]

def get_sublist_pawsx(elements, number_of_few_shots):
        sublist = []
        
        # 找到第一个 label 为 0 的元素
        for element in elements:
            if int(element['label']) == 0:
                sublist.append(element)
                break
        
        # 找到第一个 label 为 1 的元素
        for element in elements:
            if int(element['label']) == 1:
                sublist.append(element)
                break
        
        return sublist

def get_sublist_xnli(elements, number_of_few_shots):
        sublist = []
        
        # 找到第一个 label 为 0 的元素
        for element in elements:
            if element['label'] == 'not_entailment':
                sublist.append(element)
                break
        
        # 找到第一个 label 为 1 的元素
        for element in elements:
            if element['label'] == 'entailment':
                sublist.append(element)
                break
        
        return sublist

def get_sublist_mlqa(elements, number_of_few_shots):
        sublist = []
        # ipdb.set_trace()
        # # 找到第一个 label 为 0 的元素
        for element in elements:
            # if element['label'] == 'not_entailment':
            sublist.append(element)
            break
        
        # # 找到第一个 label 为 1 的元素
        # for element in elements:
        #     # if element['label'] == 'entailment':
        #     sublist.append(element)
        #     break
        
        return sublist

def get_sublist_wikiann(elements, number_of_few_shots):
        sublist = []
        
        # 找到第一个 label 为 0 的元素
        # ipdb.set_trace()
        for element in elements:
            if 'ORG' in element['label']:
                sublist.append(element)
                break
        
        # 找到第一个 label 为 1 的元素
        for element in elements:
            if 'PER' in element['label']:
                sublist.append(element)
                break

        # 找到第一个 label 为 2 的元素
        for element in elements:
            if 'LOC' in element['label']:
                sublist.append(element)
                break
        
        return sublist

MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP = {
    "gpt2-xl": 1024,
    "llama-2-7b-hf": 4096,
    "llama3-8b-instruct": 4096,
    "meta-llama-3-8b-instruct": 4096,
    "qwen2.5-7b-instruct": 4096,
    "eleutherai_gpt-j-6b": 2048,
    "gpt-j-6b": 2048,
    "gpt2-large": 1024,
    "gpt2-medium": 1024
}