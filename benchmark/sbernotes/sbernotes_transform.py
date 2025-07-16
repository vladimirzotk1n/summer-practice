from sbernotes_to_csv import transform_sbernotes_to_csv
from benchmark.csv_to_OCRBench import to_OCRBench
# Преобразуем все в dataset с колонками image и text

transform_sbernotes_to_csv("data/sbernotes/annotations_train.json",
                 "data/sbernotes/sorted_train.csv")

transform_sbernotes_to_csv("data/sbernotes/annotations_test.json",
                 "data/sbernotes/sorted_test.csv")

transform_sbernotes_to_csv("data/sbernotes/annotations_val.json",
                 "data/sbernotes/sorted_val.csv")



# Переводим в отдельные бенчмарки

train_data = pd.read_csv("data/sbernotes/sorted_train.csv")
to_OCRBench(train_data, 'data/sbernotes/sbernotes_train.json', 'sber-school_notebooks_RU')


test_data = pd.read_csv("data/sbernotes/sorted_test.csv")
to_OCRBench(test_data, 'data/sbernotes/sbernotes_test.json', 'sber-school_notebooks_RU')


val_data = pd.read_csv("data/sbernotes/sorted_val.csv")
to_OCRBench(val_data, 'data/sbernotes/sbernotes_val.json', 'sber-school_notebooks_RU')