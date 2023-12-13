import json

# FILE UTILS:
def load_jsonl(file_path):
    print('reading from', file_path)
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


# DATA UTILS
option_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}

def process_options(option):
    return option.strip("(").strip(")").strip()

def convert_alphabetical_option_to_id(ans):
    option_map = {v: k for k, v in option_map.items()}
    ans = process_options(ans)
    return option_map.get(ans, -1)

def add_pred_key(dp, key1, key2, key3):
    if key1 not in dp:
        dp[key1] = {}
    if key2 not in dp[key1]:
        dp[key1][key2] = {}
    if key3 not in dp[key1][key2]:
        dp[key1][key2][key3] = None