from collections import Counter
from data_preparation.option_sentences_preparation import get_options_string_based_on_order, get_text_with_option_words
from utils import add_pred_key, convert_alphabetical_option_to_id, load_jsonl, process_options, write_jsonl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
t5_mlm= AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

option_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}

sample_article = 'Notices of intended prosecution claimed 991 drivers had broken a 40mph (64km/h) speed limit in Conwy tunnel last October. But the limit had been imposed for night maintenance only and not lifted in the morning as it should have been. Within days the drivers got an apology in the post. North Wales Police released the figures in a Freedom of Information reply. The force said: \"The issue was caused by an administration error surrounding the enforcement period. \"North Wales police do not record the cost of cancelling notices.\"'
sample_summary = "Nearly 1,000 drivers were wrongly sent speeding notices after a @placeholder limit on a north Wales road was not lifted , figures have shown ."
sample_placeholder = "temporary"
example1 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'

sample_article = '12 June 2017 Last updated at 12:52 BST Previously code-named Project Scorpio - take a look at the new Xbox One X console. Phil Spencer, head of Xbox said it was the: \"most powerful console ever made\". The console was revealed at this year\'s E3 conference - one of the world\'s biggest gaming and technology shows. It runs from 13th to 15th June in Los Angeles, America.'
sample_summary = "Microsoft have revealed their brand - new top - @placeholder console at a big game show in America ."
sample_placeholder = "secret"
example2 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'


def all_cyclic_permutations(data):
  length = len(data)
  for i in range(length):
    yield data[i:] + data[:i]
    all_cyclic_permutations(data[1:])

def majority_vote_label(dp, outputs, option_orders):
   answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
   answer_options = [process_options(ans) for ans in answers]
   majority_label = Counter(answer_options).most_common()[0][0]
   return majority_label
   
   

def get_baseline_answer(dp, model, tokenizer, prompt_text):
    input_texts = []
    output_texts = []
    article = dp["article"]
    summary = dp["question"]

    option_orders_iter = all_cyclic_permutations(list(range(5))) if "option_5" not in dp else all_cyclic_permutations(list(range(6)))
    option_orders = [x for x in option_orders_iter]

    for option_order in option_orders:
       current_label = option_order.index(dp["label"])
       complete_prompt = f"{prompt_text} \nArticle: {article} \nSummary: {summary} \n@placeholder options {get_options_string_based_on_order(dp, option_order)} \n Answer: "
       input_texts.append(complete_prompt)
       output_texts.append(option_map[current_label])

    inputs = tokenizer(input_texts, return_tensors="pt", max_length=2048, padding = True, truncation=True)
    outputs = model.generate(input_ids=inputs["input_ids"], max_length=5)

    return majority_vote_label(dp, outputs, option_orders)


if __name__ == "__main__":

    input_file_path = "./data/baseline/dev_baseline.jsonl"
    output_file_path = "./data/baseline/dev_baseline.jsonl"
    dev_dataset = load_jsonl(input_file_path)
    model= AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    
    prompt_text = "Fill in the @placeholder with one word from the given options which best fits the @placeholder in the summary of the article"
    for dp in dev_dataset:
        answer = get_baseline_answer(dp, model, tokenizer, prompt_text)
        dp = add_pred_key(dp, "predictions", "vanilla_multi_option_ordering")
        dp["predictions"]["vanilla_multi_option"] = convert_alphabetical_option_to_id(answer)

        write_jsonl(output_file_path, dev_dataset)
    write_jsonl(output_file_path, dev_dataset)