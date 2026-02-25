import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset (example CSV)
data = pd.read_csv(r"/Users/shrutimishra/Downloads/music_genre_dataset.csv")

# Step 1: Train (70%) + Temp (30%)
train, temp = train_test_split(
    data,
    test_size=0.30,
    random_state=42,
    stratify=data["genre_label"]  # keeps genre distribution balanced
)

# Step 2: Validation (15%) + Test (15%)
val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["genre_label"]
)

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))