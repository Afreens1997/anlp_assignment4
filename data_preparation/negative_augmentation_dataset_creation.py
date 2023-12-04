from utils import load_jsonl, write_jsonl
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
t5_mlm= AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


def check():
    # Input text
    text = 'India is a <extra_id_0> of the world. </s>'

    encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
    input_ids = encoded['input_ids']

    # Generaing 20 sequences with maximum length set to 5
    outputs = t5_mlm.generate(input_ids=input_ids, 
                            num_beams=200, num_return_sequences=1,
                            max_length=5)

    _0_index = text.index('<extra_id_0>')
    _result_prefix = text[:_0_index]
    _result_suffix = text[_0_index+12:]  # 12 is the length of <extra_id_0>

    def _filter(output, end_token='<extra_id_1>'):
        # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
        _txt = t5_tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if end_token in _txt:
            _end_token_index = _txt.index(end_token)
            return _result_prefix + _txt[:_end_token_index] + _result_suffix
        else:
            return _result_prefix + _txt + _result_suffix

    results = list(map(_filter, outputs))
    print(results)


    import re

    def extract_extra_id_0_replacement(sent1, sent2):
        # Find the content corresponding to <extra_id_0> in sent2
        match = re.search(r'(.*?)<extra_id_0>', sent1)
        replacement_text = match.group(1).strip() if match else None



        # Replace <extra_id_0> with the extracted text in sent1
        result_sent1 = re.sub(r'<extra_id_0>', replacement_text, sent1)

        return replacement_text, result_sent1

    # Example usage
    sent1 = "India is a <extra_id_0> of the world. </s>"
    sent2 = "India is a big part of the world. </s>"

    output, result_sent1 = extract_extra_id_0_replacement(sent1, sent2)

    print("Extracted text:", output)
    print("Resulting sentence:", result_sent1)


    def create_negative_augmentation_dataset(tokenizer, model, question, context, prompt):
        complete_prompt = f"{prompt} {context} {question}"
        
        input_data = "The <extra_id_0> walks in the park are the only reason i go"
        input_ids = tokenizer(input_data, return_tensors="pt").input_ids

        sequence_ids = model.generate(input_ids)
        output_sequences = tokenizer.batch_decode(sequence_ids)
        print(output_sequences)

    # create_negative_augmentation_dataset(tokenizer, model, "", "", "")



def get_negative_option(tokenizer, model, prompt, article, summary):
    complete_prompt = f"{prompt} \nArticle: {article} \nSummary: {summary} \n@placeholder replacement:"
    # print(complete_prompt)
    inputs = tokenizer(complete_prompt, return_tensors="pt", max_length=2048)
    outputs = model.generate(**inputs, max_length=3)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
def create_negative_augmentation_dataset():
    train_dataset = load_jsonl("./dataset/enhanced_v1/train.jsonl")
    test_dataset = load_jsonl("./dataset/enhanced_v1/test.jsonl")
    dev_dataset = load_jsonl("./dataset/enhanced_v1/dev.jsonl")

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
            write_jsonl("./dataset/enhanced_v1/train.jsonl", train_dataset)


    for i, dp in enumerate(test_dataset):
        if "option_5" in dp:
            continue
        article = dp["article"]
        summary = dp["question"]
        option_5 = get_negative_option(t5_tokenizer, t5_mlm, prompt, article, summary)
        print(option_5)
        dp["option_5"] = option_5
        if i%50 == 0:
            write_jsonl("./dataset/enhanced_v1/train.jsonl", test_dataset)

    for i, dp in enumerate(dev_dataset):
        if "option_5" in dp:
            continue
        article = dp["article"]
        summary = dp["question"]
        option_5 = get_negative_option(t5_tokenizer, t5_mlm, prompt, article, summary)
        print(option_5)
        dp["option_5"] = option_5
        if i%50 == 0:
            write_jsonl("./dataset/enhanced_v1/train.jsonl", dev_dataset)

    write_jsonl("./dataset/enhanced_v1/train.jsonl", train_dataset)
    write_jsonl("./dataset/enhanced_v1/test.jsonl", test_dataset)
    write_jsonl("./dataset/enhanced_v1/dev.jsonl", dev_dataset)

create_negative_augmentation_dataset()
