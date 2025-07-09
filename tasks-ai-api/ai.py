from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import os
os.environ["WANDB_DISABLED"] = "true"
import torch
data = [
    ("ноутбук для работы с видео", "Apple MacBook Pro 16 M2 Pro", "Apple M2 Pro, 32GB RAM, 1TB SSD, macOS, Final Cut", "✅"),
    ("ноутбук для монтажа видео", "MSI Creator Z17", "Intel Core i9, 64GB RAM, RTX 3080, 2TB SSD", "✅"),
    ("ноутбук для видео", "ASUS ROG Zephyrus G14", "Ryzen 9 5900HS, 32GB RAM, RTX 3060, 1TB SSD", "✅"),
    ("ноутбук для видеомонтажа", "Dell XPS 15", "Intel Core i7, 32GB RAM, GTX 1650 Ti, 1TB SSD", "✅"),
    ("монтаж видео", "Lenovo Legion 5", "AMD Ryzen 7, 16GB RAM, RTX 3060, 512GB SSD", "✅"),
    ("ноутбук для работы с видео", "Apple MacBook Pro M3 Max", "Apple M3 Max, 64GB RAM, 2TB SSD, macOS", "✅"),
    ("видеомонтаж ноутбук", "Razer Blade 16", "Intel i9-13950HX, 32GB RAM, RTX 4080, SSD 2TB", "✅"),
    ("монтаж видео", "ASUS ProArt Studiobook", "Intel Xeon, 64GB RAM, RTX A5000, SSD 4TB", "✅"),
    ("редактирование видео", "Gigabyte Aero 16", "Intel i7, 32GB RAM, RTX 4070, SSD 1TB", "✅"),
    ("обработка видео", "MSI Creator M16", "Intel i7-13700H, 32GB RAM, RTX 4060, SSD 1TB", "✅"),

    ("ноутбук для работы с видео", "HP Pavilion 15", "Intel Core i5, 16GB RAM, SSD 512GB, Intel Iris Xe", "⚠️"),
    ("монтаж видео", "Acer Swift 3", "Ryzen 5, 8GB RAM, 512GB SSD, встроенная графика", "⚠️"),
    ("редактирование видео", "ASUS VivoBook 15", "Intel Core i3, 16GB RAM, SSD 256GB", "⚠️"),
    ("видеомонтаж ноутбук", "Dell Inspiron 14", "Intel Core i5, 8GB RAM, SSD 512GB, UHD Graphics", "⚠️"),
    ("ноутбук для графики и видео", "HP Envy x360", "Ryzen 7, 16GB RAM, Vega 8, SSD 1TB", "⚠️"),
    ("ноутбук для видео", "Acer Aspire 5", "Intel i5-1135G7, 8GB RAM, 512GB SSD, Intel Iris Xe", "⚠️"),
    ("видеомонтаж", "HP ProBook 450", "Intel i5, 16GB RAM, SSD 256GB, без дискретной графики", "⚠️"),
    ("монтаж видео", "Lenovo IdeaPad 5", "Ryzen 5 5500U, 8GB RAM, Vega 7, SSD 512GB", "⚠️"),
    ("обработка видео", "ASUS VivoBook 14", "Intel i3-1215U, 16GB RAM, SSD 512GB", "⚠️"),
    ("редактирование видео", "Dell Inspiron 15", "Intel i5, 8GB RAM, SSD 256GB, UHD Graphics", "⚠️"),
    
    ("ноутбук для видео", "ASUS VivoBook X512DA", "Ryzen 3 3200U, 8ГБ RAM, HDD 1ТБ, Vega 3", "❌"),
    ("монтаж видео", "Lenovo IdeaPad 3", "Intel Celeron, 4GB RAM, HDD 500GB", "❌"),
    ("ноутбук для видео", "Acer Aspire 1", "Intel Pentium N5000, 4GB RAM, 128GB eMMC", "❌"),
    ("видеомонтаж ноутбук", "Chuwi HeroBook", "Intel Atom, 4GB RAM, 64GB SSD", "❌"),
    ("обработка видео", "HP Stream", "AMD A6, 4GB RAM, 32GB eMMC", "❌"),
    ("ноутбук для работы с видео", "Lenovo V14", "Intel Core i3 1005G1, 4GB RAM, HDD 1TB, UHD Graphics", "❌"),
    ("монтаж видео", "HP 255 G7", "AMD A6-9225, 4GB RAM, HDD 500GB, Radeon R4", "❌"),
    ("обработка видео", "ASUS X543MA", "Intel Celeron N4000, 4GB RAM, 500GB HDD", "❌"),
    ("видеомонтаж ноутбук", "Acer Extensa 15", "Intel Pentium Silver, 8GB RAM, HDD 1TB", "❌"),
    ("ноутбук для видео", "Prestigio SmartBook", "Intel Atom, 4GB RAM, eMMC 64GB", "❌"),
    
    ("ноутбук для видео", "Ноутбук DEXP Aquilon", "Технические характеристики отсутствуют", "❓"),
    ("видеомонтаж ноутбук", "Irbis NB76", "Информация не указана", "❓"),
    ("ноутбук для работы", "DEXP Atlas", "Core i5, RAM неизвестна, SSD 512ГБ", "❓"),
    ("обработка видео", "DNS Office", "Неизвестный процессор, 8ГБ RAM, HDD 1ТБ", "❓"),
    ("монтаж", "Ноутбук Noname", "Данных о характеристиках нет", "❓"),
]
df = pd.DataFrame(data, columns=["query", "title", "description", "label"])
path = "laptop_relevance.csv"
df.to_csv(path, index=False)
df = pd.read_csv("laptop_relevance.csv")
df['text'] = df['query'] + " [SEP] " + df['title'] + " [SEP] " + df['description']
label_map = {"✅": 0, "⚠️": 1, "❌": 2, "❓": 3}
df['label'] = df['label'].map(label_map)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=256)
dataset = dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)
dataset = dataset.remove_columns(['query', 'title', 'description', 'text', '__index_level_0__'])
val_dataset = val_dataset.remove_columns(['query', 'title', 'description', 'text', '__index_level_0__'])

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
)

from sklearn.metrics import accuracy_score, f1_score
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
def predict(query, title, description):
    text = query + " [SEP] " + title + " [SEP] " + description
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
        pred = output.logits.argmax(dim=-1).item()
    return {0: "✅", 1: "⚠️", 2: "❌", 3: "❓"}[pred]
predict(input())# query, title, description
