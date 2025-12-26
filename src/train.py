import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from .dataset_loader import load_ravdess_dataset
from .model import LSTMEmotionModel

DATASET_PATH = "../data/ravdess"
MODEL_SAVE_PATH = "../saved_models/model.pth"

def train_model():
    print("Loading dataset...")
    X, y = load_ravdess_dataset(DATASET_PATH)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.long)

    model = LSTMEmotionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training Started...")
    epochs = 10000

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model Saved Successfully.")

    print("\nEvaluating...")
    predictions = torch.argmax(model(X_test), dim=1)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    train_model()
