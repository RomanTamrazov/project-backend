import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import os
os.environ["WANDB_DISABLED"] = "true"

labels_map = {
    0: "🌞 Позитивный",
    1: "😐 Нейтральный",
    2: "🌩️ Негативный",
    3: "🤨 Саркастичный"
}

data = [
    {"text": "Прекрасный телефон. Камера, конечно, как у микроволновки, но зато корпус блестит.", "label": 3},
    {"text": "Очень доволен покупкой! Всё работает отлично.", "label": 0},
    {"text": "Это просто ужас. Батарея не держит, греется сильно.", "label": 2},
    {"text": "Нормально, свои деньги оправдывает.", "label": 1},
    {"text": "О, какая замечательная производительность… в 2007 году, наверное.", "label": 3},
    {"text": "Купил и не пожалел. Всё быстро и удобно.", "label": 0},
    {"text": "Ну да, работает. Как и мой утюг.", "label": 3},
    {"text": "Телефон ни о чём. Зарядка медленная, экран тусклый.", "label": 2},
    {"text": "Средний аппарат. Не лучше и не хуже других.", "label": 1},
    {"text": "Просто бомба! Камера 🔥", "label": 0},
]
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_ds = Dataset.from_list(train_data)
val_ds = Dataset.from_list(val_data)

model_name = "cointegrated/rubert-tiny2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_total_limit=1,
    save_steps=500,
    logging_steps=100
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == labels)
    return {"accuracy": acc}
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()
        pred = np.argmax(probs)
    return labels_map[pred]
#-----------
print(predict_sentiment("Телефон хороший, но чувствуется дух 2010 года."))
