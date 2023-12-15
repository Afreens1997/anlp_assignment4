from data_preparation.option_sentences_preparation import get_text_with_option_words
from utils import add_pred_key, convert_alphabetical_option_to_id, load_jsonl, write_jsonl
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
        article = dp["article"]
        summary = dp["question"]
        complete_prompt = f"{prompt_text} \nArticle: {article} \nSummary: {summary} \n@placeholder options {get_text_with_option_words(dp)} \n Answer: "
        input_texts.append(complete_prompt)

    inputs = tokenizer(input_texts, return_tensors="pt", max_length=2048)
    outputs = model.generate(input_ids=inputs["input_ids"], max_length=5)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":

    input_file_path = "./data/baseline/dev_baseline.jsonl"
    output_file_path = "./data/baseline/dev_baseline.jsonl"
    dev_dataset = load_jsonl(input_file_path)
    model= AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    
    batch_size = 8
    for prompt_key, prompt_text in prompt_map.items():
        for i in range(len(dev_dataset), batch_size):
            datapoints = dev_dataset[i:i+batch_size]
            answers = get_baseline_answer(datapoints, model, tokenizer, prompt_text)
            for dp, ans in zip(datapoints, answers):
                dp = add_pred_key(dp, "predictions", "vanilla_multi_option", prompt_key)
                dp["predictions"]["vanilla_multi_option"][prompt_key] = convert_alphabetical_option_to_id(ans)

                write_jsonl(output_file_path, dev_dataset)
        write_jsonl(output_file_path, dev_dataset)