from pathlib import Path
import pandas as pd

from benchmark.csv_to_ocrbench import to_OCRBench
from sbernotes_to_csv import transform_sbernotes_to_csv
from benchmark.concatenate_benchmarks import concatenate


base_dir = Path(__file__).parent.parent.parent / "data" / "sbernotes"

train_json_path = base_dir / "annotations_train.json"
test_json_path  = base_dir / "annotations_test.json"
val_json_path   = base_dir / "annotations_val.json"

train_df_path = base_dir / "sorted_train.csv"
test_df_path  = base_dir / "sorted_test.csv"
val_df_path   = base_dir / "sorted_val.csv"

# Преобразуем все в dataset с колонками image и text
transform_sbernotes_to_csv(train_json_path, train_df_path)
transform_sbernotes_to_csv(test_json_path, test_df_path)
transform_sbernotes_to_csv(val_json_path, val_df_path)

# Переводим в отдельные бенчмарки

train_bench_path = base_dir / 'sbernotes_bench_train.json'
train_df = pd.read_csv(train_df_path)
to_OCRBench(train_df, train_bench_path, 'sber-school_notebooks_RU')

test_bench_path = base_dir / 'sbernotes_bench_test.json'
test_df = pd.read_csv(test_df_path)
to_OCRBench(test_df, test_bench_path, 'sber-school_notebooks_RU')

val_bench_path = base_dir / 'sbernotes_bench_val.json'
val_df = pd.read_csv(val_df_path)
to_OCRBench(val_df, val_bench_path, 'sber-school_notebooks_RU')

# Соединяем в один бенчмарк

concatenate(train_bench_path, test_bench_path, val_bench_path, final_bench_path="sber-bench.json")

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