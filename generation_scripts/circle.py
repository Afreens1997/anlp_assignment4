# from data_preparation.option_sentences_preparation import get_text_with_option_words
# from utils import add_pred_key, convert_alphabetical_option_to_id, load_jsonl, write_jsonl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

t5_mlm= AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


option_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F"
}
# ORDERING UTILS:

def circ_permutator(choices): # Returns ircular Permutations of a List
  perms = []
  n = len(choices)

  fronts = choices
  backs = []

  for i in range(n):
    perms.append(fronts + backs)

    pop = fronts[0]
    fronts = fronts[1:]
    backs.append(pop)
  
  return perms

def l2s(lis): # Converts List to String
  s = ""
  s = " ".join(lis)
  return s

def get_text_with_option_words(dp, include_option5 = False):
    #  Expected Output is: 
    # ['(A) w1 (B) w2 (C) w3 (D) w4 (E) w5',
    # '(B) w2 (C) w3 (D) w4 (E) w5 (A) w1',
    # '(C) w3 (D) w4 (E) w5 (A) w1 (B) w2',
    # '(D) w4 (E) w5 (A) w1 (B) w2 (C) w3',
    # '(E) w5 (A) w1 (B) w2 (C) w3 (D) w4']



    if include_option5 == False:

        options = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        words = [str(dp['option_0']), str(dp['option_1']), str(dp['option_2']), str(dp['option_3']), str(dp['option_4'])]





        # return f"(A) {dp['option_0']} (B) {dp['option_1']} (C) {dp['option_2']} (D) {dp['option_3']} (E) {dp['option_4']}"
    else:
        # return f"(A) {dp['option_0']} (B) {dp['option_1']} (C) {dp['option_2']} (D) {dp['option_3']} (E) {dp['option_4']} (F) {dp['option_5']}"

        options = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]
        words = [str(dp['option_0']), str(dp['option_1']), str(dp['option_2']), str(dp['option_3']), str(dp['option_4']), str(dp['option_5']) ]
    
    concs = [a + " " + b for a, b in zip(options, words)]
    x = [l2s(perm) for perm in circ_permutator(concs)]
    return x

import json

# FILE UTILS:
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def write_jsonl(filename, data):
    print('writing to', filename)
    with open(filename, "w+") as outfile:
        for idx, element in enumerate(data):
            # print(round(idx * 100 / len(data), 2), "%", end="\r")
            # sys.stdout.flush()
            json.dump(element, outfile)
            outfile.write("\n")

def read_json(filename):
    print('reading from', filename)
    assert filename.endswith("json"), "file provided to read_json does not end with .json extension. Please recheck!"
    return json.load(open(filename))

def write_json(data, filename):
    print('writing to', filename)
    assert filename.endswith("json"), "file provided to write_json does not end with .json extension. Please recheck!"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, sort_keys=False)


# DATA UTILS:

def process_options(option):
    return option.strip("(").strip(")").strip()

def convert_alphabetical_option_to_id(ans):
    option_map_2 = {v: k for k, v in option_map.items()}
    ans = process_options(ans)
    return option_map_2.get(ans, -1)

def add_pred_key(dp, key1, key2, key3):
    if key1 not in dp:
        dp[key1] = {}
    if key2 not in dp[key1]:
        dp[key1][key2] = {}
    if key3 not in dp[key1][key2]:
        dp[key1][key2][key3] = {}

    return dp

sample_article = 'Notices of intended prosecution claimed 991 drivers had broken a 40mph (64km/h) speed limit in Conwy tunnel last October. But the limit had been imposed for night maintenance only and not lifted in the morning as it should have been. Within days the drivers got an apology in the post. North Wales Police released the figures in a Freedom of Information reply. The force said: \"The issue was caused by an administration error surrounding the enforcement period. \"North Wales police do not record the cost of cancelling notices.\"'
sample_summary = "Nearly 1,000 drivers were wrongly sent speeding notices after a @placeholder limit on a north Wales road was not lifted , figures have shown ."
sample_placeholder = "temporary"
example1 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'

sample_article = '12 June 2017 Last updated at 12:52 BST Previously code-named Project Scorpio - take a look at the new Xbox One X console. Phil Spencer, head of Xbox said it was the: \"most powerful console ever made\". The console was revealed at this year\'s E3 conference - one of the world\'s biggest gaming and technology shows. It runs from 13th to 15th June in Los Angeles, America.'
sample_summary = "Microsoft have revealed their brand - new top - @placeholder console at a big game show in America ."
sample_placeholder = "secret"
example2 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'


prompt_map = {
    "prompt_basic" : "Fill in the @placeholder.",
    "prompt_with_task_description" : "Fill in the @placeholder with one word from the given options which best fits the @placeholder in the summary of the article",
    "prompt_with_task_description_detailed" : "The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary one of the best suited option word. Each option is an abstract word. Abstract words refer to intangible qualities, ideas, and concepts.",
    "prompt_with_task_description_1_shot" : f"Fill in the @placeholder with one word from the given options which best fits the @placeholder in the summary of the article. Example: {example1}",
    "prompt_with_task_description_2_shot" : f"Fill in the @placeholder with one word from the given options which best fits the @placeholder in the summary of the article. Example: {example1} {example2}",
}


def get_baseline_answer(batch_data, model, tokenizer, prompt_text):
    input_texts = []
    for dp in batch_data:
        # print("Hi", dp)
        # print()
        article = dp["article"]
        summary = dp["question"]
        circular_perms = get_text_with_option_words(dp)

        for circular_perm in circular_perms:
            complete_prompt = f"{prompt_text} \nArticle: {article} \nSummary: {summary} \n@placeholder options {circular_perm} \n Answer: "
            # print(complete_prompt)
            # print("_________________________________________________________________________________________________")
            input_texts.append(complete_prompt)

    inputs = tokenizer(input_texts, return_tensors="pt", max_length=2048, padding = True, truncation=True)
    outputs = model.generate(input_ids=inputs["input_ids"], max_length=5)
    a = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(len(a))
    # print(a)

    return a
    # return ["(A)" for _ in range(len(batch_data))]

if __name__ == "__main__":

    input_file_path = "/home/aaditd/anlp_assignment4/data/perm/db.jsonl"
    output_file_path = "/home/aaditd/anlp_assignment4/data/perm/db.jsonl"
    dev_dataset = load_jsonl(input_file_path)
    model= AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    
    batch_size = 1
    for prompt_key, prompt_text in prompt_map.items():
        for i in range(0, len(dev_dataset), batch_size):
            datapoints = dev_dataset[i:i+batch_size]
            answers = get_baseline_answer(datapoints, model, tokenizer, prompt_text)
            print(answers)
            print("_________________________________________________________________________________________________")
            print("_________________________________________________________________________________________________")
            print()

            
            for dp, ans in zip(datapoints, answers):
                dp = add_pred_key(dp, "predictions", "vanilla_multi_circular", prompt_key)
                # print(dp)
                dp["predictions"]["vanilla_multi_circular"][prompt_key] = ans

                write_jsonl(output_file_path, dev_dataset)
        write_jsonl(output_file_path, dev_dataset)


    print("DONE!!!")
