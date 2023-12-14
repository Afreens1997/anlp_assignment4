from data_preparation.option_sentences_preparation import get_options_string_based_on_order, get_text_with_option_words
from utils import load_jsonl, option_map, write_jsonl


prompt_text = "Fill in the @placeholder with one word from the given options which best fits the @placeholder in the summary of the article"

def all_cyclic_permutations(data):
  length = len(data)
  for i in range(length):
    yield data[i:] + data[:i]
    all_cyclic_permutations(data[1:])

def convert_dataset_to_training_format(dataset, test=False):
    training_format_dataset = []
    for i, dp in enumerate(dataset):
        print(i)
        article = dp["article"]
        summary = dp["question"]

        option_orders_iter = all_cyclic_permutations(list(range(5))) if "option_5" not in dp else all_cyclic_permutations(list(range(6)))
        option_orders = [x for x in option_orders_iter]

        for option_order in option_orders:
            complete_prompt = f"{prompt_text} \nArticle: {article} \nSummary: {summary} \n@placeholder options {get_options_string_based_on_order(dp, option_order)} \n Answer: "
            
            training_format_dataset.append({
                "input_text": complete_prompt,
                "output_text" : str(option_map[option_order.index(dp["label"])]) if not test else ""
            })

    return training_format_dataset


if __name__ == "__main__":
    negative_augmented_data_folder = "./data/dataset/original_dataset/"
    output_folder = "./data/dataset/training_format/multiple_option_with_ordering/"
    train_dataset = load_jsonl(f"{negative_augmented_data_folder}train.jsonl")
    test_dataset = load_jsonl(f"{negative_augmented_data_folder}test.jsonl")
    dev_dataset = load_jsonl(f"{negative_augmented_data_folder}dev.jsonl")

    print(train_dataset[0])
    converted_train_dataset = convert_dataset_to_training_format(train_dataset)
    converted_test_dataset = convert_dataset_to_training_format(test_dataset, test=True)
    converted_dev_dataset = convert_dataset_to_training_format(dev_dataset)

    write_jsonl(f"{output_folder}train.jsonl", converted_train_dataset)
    write_jsonl(f"{output_folder}test.jsonl", converted_test_dataset)
    write_jsonl(f"{output_folder}dev.jsonl", converted_dev_dataset)
