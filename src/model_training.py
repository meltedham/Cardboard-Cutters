# src/model_training.py

from transformers import BertForSequenceClassification, TrainingArguments, Trainer

# Step 1: Load the pre-trained BERT model for sequence classification
def load_model(num_labels):
    """
    Loads the BERT model for sequence classification.
    :param num_labels: The number of classes for classification.
    :return: Pre-trained model with classification head.
    """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return model

def setup_training_args(output_dir='./results', epochs=3, batch_size=16):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        save_strategy="epoch",   # Save the model every epoch
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        eval_steps=None,  # Evaluate every 500 steps (adjust as needed)
    )
    return training_args


# Step 3: Train the model using the Hugging Face Trainer
def train_model(model, training_args, train_dataset, eval_dataset):
    """
    Train the model using Hugging Face's Trainer API.
    :param model: The model to train.
    :param training_args: The training arguments.
    :param train_dataset: The training dataset.
    :param eval_dataset: The validation dataset.
    :return: Trained model.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()