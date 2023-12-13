from utils import load_jsonl, write_jsonl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
t5_mlm= AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


def get_negative_option(tokenizer, model, prompt, article, summary):
    complete_prompt = f"{prompt} \nArticle: {article} \nSummary: {summary} \n@placeholder replacement:"
    # print(complete_prompt)
    inputs = tokenizer(complete_prompt, return_tensors="pt", max_length=2048)
    outputs = model.generate(**inputs, max_length=3)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
def create_negative_augmentation_dataset():
    train_dataset = load_jsonl("./data/dataset/NAL/train.jsonl")
    test_dataset = load_jsonl("./data/dataset/NAL/test.jsonl")
    dev_dataset = load_jsonl("./data/dataset/NAL/dev.jsonl")

    sample_article = 'Notices of intended prosecution claimed 991 drivers had broken a 40mph (64km/h) speed limit in Conwy tunnel last October. But the limit had been imposed for night maintenance only and not lifted in the morning as it should have been. Within days the drivers got an apology in the post. North Wales Police released the figures in a Freedom of Information reply. The force said: \"The issue was caused by an administration error surrounding the enforcement period. \"North Wales police do not record the cost of cancelling notices.\"'
    sample_summary = "Nearly 1,000 drivers were wrongly sent speeding notices after a @placeholder limit on a north Wales road was not lifted , figures have shown ."
    sample_placeholder = "temporary"
    example1 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'

    sample_article = '12 June 2017 Last updated at 12:52 BST Previously code-named Project Scorpio - take a look at the new Xbox One X console. Phil Spencer, head of Xbox said it was the: \"most powerful console ever made\". The console was revealed at this year\'s E3 conference - one of the world\'s biggest gaming and technology shows. It runs from 13th to 15th June in Los Angeles, America.'
    sample_summary = "Microsoft have revealed their brand - new top - @placeholder console at a big game show in America ."
    sample_placeholder = "secret"
    example2 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'

    prompt = f"The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary with only one appropriate abstract word. Abstract words refer to intangible qualities, ideas, and concepts. For example: \n{example1} \n{example2}\n"
    
    for i, dp in enumerate(train_dataset):
        # if "option_5" in dp:
        #     continue
        article = dp["article"]
        summary = dp["question"]
        option_5 = get_negative_option(t5_tokenizer, t5_mlm, prompt, article, summary)
        print(option_5)
        dp["option_5"] = option_5
        if i%50 == 0:
            write_jsonl("./data/dataset/NAL/train.jsonl", train_dataset)


    for i, dp in enumerate(test_dataset):
        if "option_5" in dp:
            continue
        article = dp["article"]
        summary = dp["question"]
        option_5 = get_negative_option(t5_tokenizer, t5_mlm, prompt, article, summary)
        print(option_5)
        dp["option_5"] = option_5
        if i%50 == 0:
            write_jsonl("./data/dataset/NAL/train.jsonl", test_dataset)

    for i, dp in enumerate(dev_dataset):
        if "option_5" in dp:
            continue
        article = dp["article"]
        summary = dp["question"]
        option_5 = get_negative_option(t5_tokenizer, t5_mlm, prompt, article, summary)
        print(option_5)
        dp["option_5"] = option_5
        if i%50 == 0:
            write_jsonl("./data/dataset/NAL/train.jsonl", dev_dataset)

    write_jsonl("./data/dataset/NAL/train.jsonl", train_dataset)
    write_jsonl("./data/dataset/NAL/test.jsonl", test_dataset)
    write_jsonl("./data/dataset/NAL/dev.jsonl", dev_dataset)

create_negative_augmentation_dataset()
