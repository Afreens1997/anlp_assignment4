from utils import load_jsonl, process_options, option_map



def accuracy(data, answer_lambda=lambda x:x["answer"]):
    correct_count = 0
    for dp in data:
        if isinstance(answer_lambda(dp), int):
            if dp['label'] == answer_lambda(dp):
                correct_count+=1
        else:
            if option_map[dp['label']] == process_options(answer_lambda(dp)):
                correct_count+=1
    acc = round(float(correct_count/len(data)), 3)
    return acc

if __name__ == "__main__":
    print("prompt_basic", accuracy(load_jsonl("./dataset/baseline/dev.jsonl"), lambda x:x["baseline_answers"]["prompt_basic"]))
    print("prompt_with_task_description", accuracy(load_jsonl("./dataset/baseline/dev.jsonl"), lambda x:x["baseline_answers"]["prompt_with_task_description"]))
    print("prompt_1_shot", accuracy(load_jsonl("./dataset/baseline/dev.jsonl"), lambda x:x["baseline_answers"]["prompt_1_shot"]))
    print("prompt_2_shot", accuracy(load_jsonl("./dataset/baseline/dev.jsonl"), lambda x:x["baseline_answers"]["prompt_2_shot"]))
    print("prompt_basic", accuracy(load_jsonl("./dataset/baseline/dev_max_length_5.jsonl"), lambda x:x["baseline_answers"]["prompt_basic"]))
    print("prompt_basic", accuracy(load_jsonl("./dataset/baseline/dev_instruction.jsonl"), lambda x:x["baseline_answers"]["prompt_basic"]))
    print("prompt_basic", accuracy(load_jsonl("./dataset/baseline/dev_instruction_large.jsonl"), lambda x:x["baseline_answers"]["prompt_basic"]))



