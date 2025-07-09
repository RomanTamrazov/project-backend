from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("laptop_relevance.csv")  # query, title, description, label
df['text'] = df['query'] + " [SEP] " + df['title'] + " [SEP] " + df['description']
label_map = {"✅": 0, "⚠️": 1, "❌": 2, "❓": 3}
df['label'] = df['label'].map(label_map)
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
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
    output = model(**tokens)
    pred = output.logits.argmax(dim=-1).item()
    return {0: "✅", 1: "⚠️", 2: "❌", 3: "❓"}[pred]

predict("ноутбук для работы с видео", "ASUS VivoBook X512DA", "AMD Ryzen 3, 8ГБ, HDD")
