import json

def concatenate(json_train_path, json_test_path, json_val_path, final_bench_path):
    with open(json_train_path, "r", encoding="utf-8") as f1:
        data1 = json.load(f1)

    with open(json_test_path, "r", encoding="utf-8") as f2:
        data2 = json.load(f2)

    with open(json_val_path, "r", encoding="utf-8") as f1:
        data3 = json.load(f1)

    merged = data1 + data2 + data3

    with open(final_bench_path, "w", encoding="utf-8") as out_file:
        json.dump(merged, out_file, ensure_ascii=False, indent=4)

