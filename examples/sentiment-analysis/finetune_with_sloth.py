from sloth import SlothModel, SlothTrainer, SlothTokenizer

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = SlothTokenizer.from_pretrained(model_name)
model = SlothModel.from_pretrained(model_name, num_labels=2)

# Load and tokenize your dataset
train_texts = ["I love this!", "I hate this!"]
train_labels = [1, 0]
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")

# Initialize the Sloth trainer with default settings
trainer = SlothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_encodings,
    output_dir="./results"
)

# Fine-tune the model
trainer.train()
