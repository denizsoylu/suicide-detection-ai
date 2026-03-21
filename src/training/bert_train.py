import os
import torch
import pandas as pd

from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from training.bert_dataset import SuicideDataset


# ===============================
# 1. DEVICE (CPU / GPU)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)


# ===============================
# 2. DATA LOADING
# ===============================
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(
        base_dir, "data", "processed", "cleaned_suicide_data.csv"
    )

    print(f"Veri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)

    # Güvenlik
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    # EN ÖNEMLİ KISIM (VERİYİ KÜÇÜLTÜYORUZ)
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    print("Kullanılan örnek sayısı:", len(df))

    return df


# ===============================
# 3. DATALOADER OLUŞTURMA
# ===============================
def create_dataloaders(df, tokenizer, batch_size=8):
    texts = df["text"].values
    labels = df["class"].map({
        "non-suicide": 0,
        "suicide": 1
    }).values

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_dataset = SuicideDataset(X_train, y_train, tokenizer)
    val_dataset = SuicideDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


# ===============================
# 4. MODEL
# ===============================
def load_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    model.to(device)
    return model


# ===============================
# 5. OPTIMIZER
# ===============================
def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-5)


# ===============================
# 6. TRAIN ONE EPOCH
# ===============================
def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔹 Canlı ilerleme göstergesi
        if step % 50 == 0:
            print(f"Batch {step}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


# ===============================
# 7. EVALUATION
# ===============================
def evaluate(model, dataloader):
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))


# ===============================
# 8. MAIN
# ===============================
def main():
    df = load_data()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader, val_loader = create_dataloaders(df, tokenizer)

    model = load_model()
    optimizer = get_optimizer(model)

    # SADECE 1 EPOCH
    epochs = 1
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1} / {epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer)
        print(f"Train Loss: {train_loss:.4f}")

        evaluate(model, val_loader)


if __name__ == "__main__":
    main()
