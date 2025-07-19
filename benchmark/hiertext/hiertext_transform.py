from hiertext_to_csv import transform_hiertext_to_csv, transform_hiertext_to_csv_sorted
import csv
from pathlib import Path
import pandas as pd
from benchmark.csv_to_ocrbench import to_OCRBench
from benchmark.concatenate_benchmarks import concatenate

base_dir = Path(__file__).parent.parent.parent / "data" / "hiertext"

train_json_path = base_dir / "gt" / "train.jsonl"
test_json_path  = base_dir / "gt" / "test.jsonl"
val_json_path   = base_dir / "gt" / "validation.jsonl"

train_image_path = base_dir / "train"
test_image_path = base_dir / "test"
val_image_path = base_dir / "validation"

train_df_path = base_dir / "sorted_train.csv"
test_df_path  = base_dir / "sorted_test.csv"
val_df_path   = base_dir / "sorted_val.csv"

# Преобразуем все в dataset с колонками image и text

transform_hiertext_to_csv(train_json_path, train_image_path, train_df_path)
transform_hiertext_to_csv(test_json_path, test_image_path, test_df_path)
transform_hiertext_to_csv(val_json_path, val_image_path, val_df_path)

# Переводим в отдельные бенчмарки

train_bench_path = base_dir / "hiertext_bench_train.json"
train_df = pd.read_csv(train_df_path)
to_OCRBench(train_df, train_bench_path, "hiertext")

test_bench_path = base_dir / "hiertext_bench_test.json"
test_df = pd.read_csv(test_df_path)
to_OCRBench(test_df, test_bench_path, "hiertext")

val_bench_path = base_dir / "hiertext_bench_val.json"
val_df = pd.read_csv(val_df_path)
to_OCRBench(val_df, val_bench_path, "hiertext")

# Соединяем в один бенчмарк

concatenate(train_bench_path, test_bench_path, val_bench_path, final_bench_path="hiertext-bench.json")

# Очищаем от промежуточных данных
to_delete = [
    train_df_path,
    test_df_path,
    val_df_path,
    train_bench_path,
    test_bench_path,
    val_bench_path,
]

for path in to_delete:
    if path.exists():
        path.unlink()
