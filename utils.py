import json

def load_jsonl(file_path):
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
            # print(round(idx * 100 / len(data), 2), "%", end="\r")
            # sys.stdout.flush()
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
