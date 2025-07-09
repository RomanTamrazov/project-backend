import os
os.environ["WANDB_DISABLED"] = "true"

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
data = [
    ("ноутбук для работы с видео", "Apple MacBook Pro 16 M2 Pro",
     "Apple M2 Pro", "32GB", "1TB SSD", "macOS, Final Cut", "✅"),

    ("ноутбук для монтажа видео", "MSI Creator Z17",
     "Intel Core i9", "64GB", "2TB SSD", "RTX 3080", "✅"),

    ("ноутбук для видео", "ASUS ROG Zephyrus G14",
     "Ryzen 9 5900HS", "32GB", "1TB SSD", "RTX 3060", "✅"),

    ("ноутбук для видеомонтажа", "Dell XPS 15",
     "Intel Core i7", "32GB", "1TB SSD", "GTX 1650 Ti", "✅"),

    ("монтаж видео", "Lenovo Legion 5",
     "AMD Ryzen 7", "16GB", "512GB SSD", "RTX 3060", "✅"),

    ("ноутбук для работы с видео", "Apple MacBook Pro M3 Max",
     "Apple M3 Max", "64GB", "2TB SSD", "macOS", "✅"),

    ("видеомонтаж ноутбук", "Razer Blade 16",
     "Intel i9-13950HX", "32GB", "2TB SSD", "RTX 4080", "✅"),

    ("монтаж видео", "ASUS ProArt Studiobook",
     "Intel Xeon", "64GB", "4TB SSD", "RTX A5000", "✅"),

    ("редактирование видео", "Gigabyte Aero 16",
     "Intel i7", "32GB", "1TB SSD", "RTX 4070", "✅"),

    ("обработка видео", "MSI Creator M16",
     "Intel i7-13700H", "32GB", "1TB SSD", "RTX 4060", "✅"),

    ("ноутбук для работы с видео", "HP Pavilion 15",
     "Intel Core i5", "16GB", "512GB SSD", "Intel Iris Xe", "⚠️"),

    ("монтаж видео", "Acer Swift 3",
     "Ryzen 5", "8GB", "512GB SSD", "встроенная графика", "⚠️"),

    ("редактирование видео", "ASUS VivoBook 15",
     "Intel Core i3", "16GB", "256GB SSD", "", "⚠️"),

    ("видеомонтаж ноутбук", "Dell Inspiron 14",
     "Intel Core i5", "8GB", "512GB SSD", "UHD Graphics", "⚠️"),

    ("ноутбук для графики и видео", "HP Envy x360",
     "Ryzen 7", "16GB", "1TB SSD", "Vega 8", "⚠️"),

    ("ноутбук для видео", "Acer Aspire 5",
     "Intel i5-1135G7", "8GB", "512GB SSD", "Intel Iris Xe", "⚠️"),

    ("видеомонтаж", "HP ProBook 450",
     "Intel i5", "16GB", "256GB SSD", "без дискретной графики", "⚠️"),

    ("монтаж видео", "Lenovo IdeaPad 5",
     "Ryzen 5 5500U", "8GB", "512GB SSD", "Vega 7", "⚠️"),

    ("обработка видео", "ASUS VivoBook 14",
     "Intel i3-1215U", "16GB", "512GB SSD", "", "⚠️"),

    ("редактирование видео", "Dell Inspiron 15",
     "Intel i5", "8GB", "256GB SSD", "UHD Graphics", "⚠️"),

    ("ноутбук для видео", "ASUS VivoBook X512DA",
     "Ryzen 3 3200U", "8GB", "1TB HDD", "Vega 3", "❌"),

    ("монтаж видео", "Lenovo IdeaPad 3",
     "Intel Celeron", "4GB", "500GB HDD", "", "❌"),

    ("ноутбук для видео", "Acer Aspire 1",
     "Intel Pentium N5000", "4GB", "128GB eMMC", "", "❌"),

    ("видеомонтаж ноутбук", "Chuwi HeroBook",
     "Intel Atom", "4GB", "64GB SSD", "", "❌"),

    ("обработка видео", "HP Stream",
     "AMD A6", "4GB", "32GB eMMC", "", "❌"),

    ("ноутбук для работы с видео", "Lenovo V14",
     "Intel Core i3 1005G1", "4GB", "1TB HDD", "UHD Graphics", "❌"),

    ("монтаж видео", "HP 255 G7",
     "AMD A6-9225", "4GB", "500GB HDD", "Radeon R4", "❌"),

    ("обработка видео", "ASUS X543MA",
     "Intel Celeron N4000", "4GB", "500GB HDD", "", "❌"),

    ("видеомонтаж ноутбук", "Acer Extensa 15",
     "Intel Pentium Silver", "8GB", "1TB HDD", "", "❌"),

    ("ноутбук для видео", "Prestigio SmartBook",
     "Intel Atom", "4GB", "64GB eMMC", "", "❌"),

    ("ноутбук для видео", "Ноутбук DEXP Aquilon",
     "", "", "", "", "❓"),

    ("видеомонтаж ноутбук", "Irbis NB76",
     "", "", "", "", "❓"),

    ("ноутбук для работы", "DEXP Atlas",
     "Core i5", "RAM неизвестна", "512GB SSD", "", "❓"),

    ("обработка видео", "DNS Office",
     "Неизвестный процессор", "8GB", "1TB HDD", "", "❓"),

    ("монтаж", "Ноутбук Noname",
     "", "", "", "", "❓"),
]

df = pd.DataFrame(data, columns=["query", "title", "cpu", "ram", "storage", "gpu", "label"])
df['text'] = df['query'] + " [SEP] " + df['title'] + " [SEP] " + df['cpu'] + " " + df['ram'] + " " + df['storage'] + " " + df['gpu']
label_map = {"✅": 0, "⚠️": 1, "❌": 2, "❓": 3}
df['label'] = df['label'].map(label_map)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=256)
dataset = dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

remove_cols = ['query', 'title', 'cpu', 'ram', 'storage', 'gpu', 'text', '__index_level_0__']
dataset = dataset.remove_columns([c for c in remove_cols if c in dataset.column_names])
val_dataset = val_dataset.remove_columns([c for c in remove_cols if c in val_dataset.column_names])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    metric_for_best_model="accuracy",
    seed=42
)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted')
    }
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
def predict(query, title, cpu, ram, storage, gpu):
    text = query + " [SEP] " + title + " [SEP] " + cpu + " " + ram + " " + storage + " " + gpu
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
        pred = output.logits.argmax(dim=-1).item()
    return {0: "✅", 1: "⚠️", 2: "❌", 3: "❓"}[pred]
#---------
print(predict(
    "ноутбук для работы с видео",
    "Apple MacBook Pro 16 M2 Pro",
    "Apple M2 Pro",
    "32GB",
    "1TB SSD",
    "macOS, Final Cut"
))#query, title, cpu, ram, storage, gpu
