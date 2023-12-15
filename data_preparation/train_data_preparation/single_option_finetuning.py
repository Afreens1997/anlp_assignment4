from data_preparation.option_sentences_preparation import get_text_with_option_words
from utils import load_jsonl, option_map, write_jsonl


prompt_text = "Is the summary provided a correct and coherent summary of the article?"

def get_model_input(dp):
    article = dp["article"]
    summary = dp["question"]
    inputs = []
    outputs = []
    for option in range(5):
        answer = "yes" if option == dp["label"] else "no"
        inputs.append(f"{prompt_text} \nArticle: {article} \nSummary: {summary.replace('@placeholder', dp[f'option_{option}'])} Answer: ")
        outputs.append(answer)

    return inputs, outputs

def convert_dataset_to_training_format(dataset, test=False):
    training_format_dataset = []
    for i, dp in enumerate(dataset):
        
        inputs, outputs = get_model_input(dp)
        for i,o in zip(inputs, outputs):
            training_format_dataset.append({
                "input_text": i,
                "output_text" : o
            })

    return training_format_dataset


if __name__ == "__main__":
    negative_augmented_data_folder = "./data/dataset/original_dataset/"
    output_folder = "./data/dataset/training_format/single_option/"
    train_dataset = load_jsonl(f"{negative_augmented_data_folder}train.jsonl")
    test_dataset = load_jsonl(f"{negative_augmented_data_folder}test.jsonl")
    dev_dataset = load_jsonl(f"{negative_augmented_data_folder}dev.jsonl")

    print(train_dataset[0])
    converted_train_dataset = convert_dataset_to_training_format(train_dataset)
    converted_dev_dataset = convert_dataset_to_training_format(dev_dataset)

    write_jsonl(f"{output_folder}train.jsonl", converted_train_dataset)
    write_jsonl(f"{output_folder}dev.jsonl", converted_dev_dataset)
