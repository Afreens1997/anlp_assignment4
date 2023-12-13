# Functions that take in a datapoint and return the content in processed format for input to the model

import math
import torch.nn.functional as F

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from projects.anlp_assignment4.utils import add_pred_key, load_jsonl, write_jsonl


ARTICLE = "article"
SUMMARY = "question"
EXTRA_ID_TOKEN_INDEX = 32099
TOTAL_OPTIONS = 5

def single_option_generation_v1(dp, model, tokenizer):
    """
    Function takes in a dp, model and tokenizer and return the option with yields the maximum text log probability. Here, text log probability is the sum od log prob of each token in the output expected text after replacing @placeholder with option_word.
    input text to model --> f"Fill in the blank: Article: {article} \nSummary: {summary.replace('@placeholder', '<extra_id_0>')}"
    output expected text from model --> f"Fill in the blank: Article: {article} \nSummary: {summary.replace('@placeholder', option_word)}"
    return option with max output text log prob
    """
    article = dp[ARTICLE]
    summary = dp[SUMMARY]

    option_log_prob = []

    model_input_text = f"Fill in the blank: Article: {article} \nSummary: {summary.replace('@placeholder', '<extra_id_0>')}"

    #encode inputs
    input_sequences = [model_input_text]*TOTAL_OPTIONS
    complete_prompt_input_ids = tokenizer(input_sequences, return_tensors="pt", max_length=2048).input_ids

    #encode outputs
    output_sequences = []
    for i in range(TOTAL_OPTIONS):
        option_id = f"option_{i}"
        option_word = dp[option_id]
        output_sequences.append(f"Fill in the blank: Article: {article} \nSummary: {summary.replace('@placeholder', option_word)}")
    complete_prompt_output_ids = tokenizer(output_sequences, return_tensors="pt", max_length=2048).input_ids

    # get model outputs
    outputs = model(input_ids=complete_prompt_input_ids, decoder_input_ids=complete_prompt_output_ids)
    probs = outputs.logits.softmax(-1).detach().cpu()

    for option in range(TOTAL_OPTIONS):
        sentence_log_probs = []
        for i in range(len(complete_prompt_output_ids)):
            p = probs[option,i, complete_prompt_output_ids[i]].item()
            sentence_log_probs.append(math.log(p))
        option_log_prob[i] = sum(sentence_log_probs)

    return option_log_prob.index(max(option_log_prob))


def single_option_generation_v2(dp, model, tokenizer):
    """
    Function takes in a dp, model and tokenizer and return the option with yields the maximum text log probability. Here, text log probability is the sum od log prob of each token in the output expected text after replacing @placeholder with option_word.
    input text to model --> f"Fill in the blank: Article: {article} \nSummary: {summary.replace('@placeholder', '<extra_id_0>')}"
    output expected text from model --> f"Fill in the blank: Article: {article} \nSummary: {summary.replace('@placeholder', option_word)}"
    return option with max log prob to option token.
    """
    article = dp[ARTICLE]
    summary = dp[SUMMARY]

    option_log_prob = []

    model_input_text = f"Fill in the blank: Article: {article} \nSummary: {summary.replace('@placeholder', '<extra_id_0>')}"

    #encode inputs
    input_sequences = [model_input_text]*TOTAL_OPTIONS
    complete_prompt_input_ids = tokenizer(input_sequences, return_tensors="pt", max_length=2048).input_ids

    #encode outputs
    output_sequences = []
    for i in range(TOTAL_OPTIONS):
        option_id = f"option_{i}"
        option_word = dp[option_id]
        output_sequences.append(f"Fill in the blank: Article: {article} \nSummary: {summary.replace('@placeholder', option_word)}")
    complete_prompt_output_ids = tokenizer(output_sequences, return_tensors="pt", max_length=2048).input_ids

    placeholder_index = list(complete_prompt_input_ids[0]).index(EXTRA_ID_TOKEN_INDEX)
    predicted_word_id = complete_prompt_output_ids[0][placeholder_index]

    # get model outputs
    outputs = model(input_ids=complete_prompt_input_ids, decoder_input_ids=complete_prompt_output_ids)
    for option in range(TOTAL_OPTIONS):
        logits = outputs.logits[option]
        predicted_word_logit = logits[placeholder_index]
        probabilities = F.softmax(predicted_word_logit, dim=-1)
        output_word_probability = probabilities[predicted_word_id]
        option_log_prob.append(output_word_probability)

    return option_log_prob.index(max(option_log_prob))

if __name__ == "__main__":

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    dev_dataset = load_jsonl("./data/baseline/dev_baseline.jsonl")
    correct_v1 = 0
    correct_v2 = 0
    for dp in dev_dataset:
        
        predicted_option = single_option_generation_v1(dp, model, tokenizer)
        dp = add_pred_key(dp, "predictions", "vanilla_single_option", "single_option_generation_v1")
        dp["predictions"]["vanilla_single_option"]["single_option_generation_v1"] = predicted_option
        if predicted_option == dp["label"]:
            correct_v1+=1

        predicted_option = single_option_generation_v2(dp, model, tokenizer)
        dp = add_pred_key(dp, "predictions", "vanilla_single_option", "single_option_generation_v2")
        dp["predictions"]["vanilla_single_option"]["single_option_generation_v2"] = predicted_option
        if predicted_option == dp["label"]:
            correct_v2+=1

    accuracy_v1 = correct_v1/len(dev_dataset)
    accuracy_v2 = correct_v2/len(dev_dataset)

    print(accuracy_v1)
    print(accuracy_v2)

    write_jsonl("./data/baseline/dev_baseline.jsonl", dev_dataset)



        