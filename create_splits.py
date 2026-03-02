import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("auth_full_dataset.csv")

# 70% Train, 30% Temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# 15% Val, 15% Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

train_df.to_csv("train_auth_final.csv", index=False)
val_df.to_csv("val_auth_final.csv", index=False)
test_df.to_csv("test_auth_final.csv", index=False)

print("✅ Train/Val/Test splits created")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))