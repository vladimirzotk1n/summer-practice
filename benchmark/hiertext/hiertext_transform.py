from hiertext_to_csv import extract_hiertext_with_order
import csv

pairs = extract_hiertext_with_order("C:\\hiertext\\gt\\train.jsonl", 'C:\\hiertext\\train')

with open("hiertext_train.csv", "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "text"])
    for filename, text in pairs:
        writer.writerow([filename, text])



# Доделать
