### Структура проекта


```
│
├── benchmark/                  ← Работа с бенчмарками
│   ├── __init__.py             ← Инициализация модуля
│   ├── concatenate_benchmarks.py   ← Объединение бенчмарков
│   ├── csv_to_ocrbench.py      ← Конвертация CSV → OCR-бенчмарк
│   ├── sorting_simple.py       ← Простая сортировка bounding box'ов
│   │
│   ├── hiertext/               ← Обработка бенчмарка HierText
│   │   ├── hiertext-bench.json ← Финальный бенчмарк в унифицированном формате
│   │   ├── hiertext_to_csv.py  ← Конвертация HierText → CSV
│   │   └── hiertext_transform.py ← Трансформация данных HierText
│   │
│   └── sbernotes/              ← Обработка бенчмарка SberNotes
│       ├── sber-bench.json     ← Финальный бенчмарк в унифицированном формате
│       ├── sbernotes_to_csv.py ← Конвертация SberNotes → CSV
│       └── sbernotes_transform.py ← Трансформация данных SberNotes
│
├── data/                       ← Исходные данные (в .gitignore)
│   ├── hiertext/               ← Исходные данные HierText
│   └── sbernotes/              ← Исходные данные SberNotes
│
├── evaluation/                 ← Оценка качества ответов модели
│   ├── __init__.py             ← Инициализация модуля
│   ├── evaluate.py             ← Основной скрипт оценки
│   ├── metrics.py              ← Реализация метрик (F1, NED, CER)
│   ├── ollama.py               ← Код предсказания моделью
│   ├── scores-hiertext.json    ← Результаты оценки на HierText
│   └── scores-sbernotes.json   ← Результаты оценки на SberNotes
│
├── plots.ipynb                 ← Jupyter Notebook с визуализацией результатов
├── README.md                   ← Описание проекта
└── requirements.txt            ← Зависимости Python 
```
