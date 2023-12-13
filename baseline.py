from data_preparation.option_sentences_preparation import get_text_with_option_words
from utils import load_jsonl, write_jsonl
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
t5_mlm= AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

option_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}

def get_baseline_answer(tokenizer, model, prompt, dp):
    article = dp["article"]
    summary = dp["question"]
    complete_prompt = f"{prompt} \nArticle: {article} \nSummary: {summary} \n@placeholder options {get_text_with_option_words(dp)} \n @placeholder replacement:"
    # print(complete_prompt)
    inputs = tokenizer(complete_prompt, return_tensors="pt", max_length=2048)
    outputs = model.generate(**inputs, max_length=5)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def baseline_generation():
    # train_dataset = load_jsonl("./dataset/train.jsonl")
    # test_dataset = load_jsonl("./dataset/test.jsonl")
    dev_dataset = load_jsonl("./data/baseline/dev_instruction_large.jsonl")

    sample_article = 'Notices of intended prosecution claimed 991 drivers had broken a 40mph (64km/h) speed limit in Conwy tunnel last October. But the limit had been imposed for night maintenance only and not lifted in the morning as it should have been. Within days the drivers got an apology in the post. North Wales Police released the figures in a Freedom of Information reply. The force said: \"The issue was caused by an administration error surrounding the enforcement period. \"North Wales police do not record the cost of cancelling notices.\"'
    sample_summary = "Nearly 1,000 drivers were wrongly sent speeding notices after a @placeholder limit on a north Wales road was not lifted , figures have shown ."
    sample_placeholder = "temporary"
    example1 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'

    sample_article = '12 June 2017 Last updated at 12:52 BST Previously code-named Project Scorpio - take a look at the new Xbox One X console. Phil Spencer, head of Xbox said it was the: \"most powerful console ever made\". The console was revealed at this year\'s E3 conference - one of the world\'s biggest gaming and technology shows. It runs from 13th to 15th June in Los Angeles, America.'
    sample_summary = "Microsoft have revealed their brand - new top - @placeholder console at a big game show in America ."
    sample_placeholder = "secret"
    example2 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'

    prompt_basic = f"Fill in the @placeholder with one word from the given options which best fits the @placeholder in the summary of the article?"
    prompt_2_shot = f"The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary one of the best suited option word. Each option is an abstract word. Abstract words refer to intangible qualities, ideas, and concepts. For example: \n{example1} \n{example2}\n"
    # prompt_with_task_description = f"The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary one of the best suited option word. Each option is an abstract word. Abstract words refer to intangible qualities, ideas, and concepts."
    # prompt_1_shot = f"The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary one of the best suited option word. Each option is an abstract word. Abstract words refer to intangible qualities, ideas, and concepts. For example: \n{example1}"
    # prompt_2_shot = f"The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary one of the best suited option word. Each option is an abstract word. Abstract words refer to intangible qualities, ideas, and concepts. For example: \n{example1} \n{example2}\n"

    for i, dp in enumerate(dev_dataset):
        # if "baseline_answers" in dp:
        #     continue
        
        dp["baseline_answers"] = {}
        
        answer = get_baseline_answer(t5_tokenizer, t5_mlm, prompt_basic, dp)
        dp["baseline_answers"]["prompt_basic"] = answer

        # answer = get_baseline_answer(t5_tokenizer, t5_mlm, prompt_with_task_description, dp)
        # dp["baseline_answers"]["prompt_with_task_description"] = answer

        # answer = get_baseline_answer(t5_tokenizer, t5_mlm, prompt_1_shot, dp)
        # dp["baseline_answers"]["prompt_1_shot"] = answer

        # answer = get_baseline_answer(t5_tokenizer, t5_mlm, prompt_2_shot, dp)
        # dp["baseline_answers"]["prompt_2_shot"] = answer
        print(i, dp["baseline_answers"], option_map[dp["label"]])
        if i%50 == 0:
            write_jsonl("./data/baseline/dev_instruction_large.jsonl", dev_dataset)

    write_jsonl("./data/baseline/dev_instruction_large.jsonl", dev_dataset)

baseline_generation()