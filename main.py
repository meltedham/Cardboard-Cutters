# main.py

from src.data_preprocessing import load_data, tokenize_data, split_data
from src.model_training import load_model, setup_training_args, train_model
from transformers import BertTokenizer

# Step 1: Load data
df = load_data("data/reviews.csv")

# Step 2: Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenize_data(df[:80], tokenizer)  # 80% for training
val_encodings = tokenize_data(df[80:], tokenizer)    # 20% for validation

# Step 3: Split data into train and validation sets (already done)
train_df, val_df = split_data(df)

# Step 4: Load the pre-trained BERT model
model = load_model(num_labels=4)  # Assume 4 categories

# Step 5: Set up the training arguments
training_args = setup_training_args()

# Step 6: Train the model
trainer = train_model(model, training_args, train_encodings, val_encodings)

# Step 7: Save the trained model
trainer.save_model("./final_model")

