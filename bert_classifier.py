#!/usr/bin/env python
# coding: utf-8

# In[38]:


get_ipython().system('pip install openpyxl')

# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support,  classification_report

from datasets import Dataset as HFDataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import matplotlib.pyplot as plt
from evaluate import load as load_metric
from imblearn.over_sampling import RandomOverSampler
import os
import pickle
import optuna
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, EvalPrediction
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import Dataset


# In[39]:


# Set environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Helps with fragmentation
torch.cuda.empty_cache()

# Enable cuDNN for optimized training
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Check if CUDA is available
print(torch.cuda.is_available())  # Should return True if a GPU is available
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.get_device_name(0))  # Name of the GPU


# In[40]:


# ===============================
# Config
# ===============================
LABEL_COLUMN = 'CORRECT MAPPING'
TEXT_COLUMN = 'combined_text'
MIN_SAMPLES_PER_CLASS = 5
DATA_PATH = r'C:\Users\BaranMoradkhani\Desktop\project\OutPut_Top1_Corrected.xlsx' 
ENCODER_PATH = 'C:\\Users\\BaranMoradkhani\\Desktop\\project\\Corrected_label_encoder.pkl'

# ===============================
# Step 1: Load & Collapse Labels
# ===============================
def load_and_prepare_data(path, label_column, min_samples=5):
    df = pd.read_excel(path)
    df[label_column] = df[label_column].fillna('Not Applicable')
    df['combined_text'] = df['Material Name'].astype(str)
    label_counts = df[label_column].value_counts()
    rare_labels = label_counts[label_counts < min_samples].index
    df[label_column] = df[label_column].apply(lambda x: 'Other' if x in rare_labels else x)
    return df


# In[41]:


# ===============================
# Step 2: Encode Labels
# ===============================
def encode_labels(df, label_column, encoder_path):
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df[label_column])
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    return df, encoder


# In[42]:


# ===============================
# Step 3: Split & Balance
# ===============================
def split_and_balance(df):
    X = df[['combined_text']]  # Make sure it's a DataFrame
    y = df['label']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

    # Oversample
    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

    return X_train_bal['combined_text'], y_train_bal, X_val['combined_text'], y_val, X_test['combined_text'], y_test
    
def plot_label_distribution(y_train, y_val, y_test, encoder):
    plt.figure(figsize=(15, 4))
    for i, (y, title) in enumerate(zip([y_train, y_val, y_test], ["Train", "Validation", "Test"])):
        plt.subplot(1, 3, i+1)
        class_counts = pd.Series(y).map(dict(enumerate(encoder.classes_))).value_counts()
        class_counts.plot(kind='bar')
        plt.title(f"{title} Set Label Distribution")
        plt.xticks(rotation=90)
        plt.tight_layout()
    plt.show()


# In[43]:


# ===============================
# Step 4: Tokenize
# ===============================
def tokenize_data(texts, labels, tokenizer):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)
    return {
        'input_ids': torch.tensor(encodings['input_ids']),
        'attention_mask': torch.tensor(encodings['attention_mask']),
        'labels': torch.tensor(labels.tolist())
    }

class MaterialDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


# In[44]:


# ===============================
# Step 5: Weighted BERT Model
# ===============================
class WeightedBERT(torch.nn.Module):
    def __init__(self, num_labels, weights):
        super().__init__()
        #self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=num_labels)
         # Freeze all except last 4 layers
        for name, param in self.bert.bert.named_parameters():
            if any(f'layer.{i}' in name for i in range(6, 12)):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {'loss': loss, 'logits': logits}


# In[45]:


# ===============================
# Step 6: Metrics
# ===============================
def compute_metrics(eval_pred: EvalPrediction):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# In[46]:


from transformers import TrainerCallback

class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"📊 Epoch {state.epoch}: {logs}")


# In[47]:


# ===============================
# Main Training Pipeline
# ===============================
def train_pipeline():
    # Load and prep data
    df = load_and_prepare_data(DATA_PATH, LABEL_COLUMN, MIN_SAMPLES_PER_CLASS)
    df, encoder = encode_labels(df, LABEL_COLUMN, ENCODER_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = split_and_balance(df)
    plot_label_distribution(y_train, y_val, y_test, encoder)

    # Tokenize
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
   
    train_data = tokenize_data(X_train, y_train, tokenizer)
    val_data = tokenize_data(X_val, y_val, tokenizer)
    test_data = tokenize_data(X_test, y_test, tokenizer)

    train_dataset = MaterialDataset(train_data)
    val_dataset = MaterialDataset(val_data)
    test_dataset = MaterialDataset(test_data)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Model
    model = WeightedBERT(num_labels=len(encoder.classes_), weights=class_weights_tensor).to(device)
    # Patch Trainer._save to make tensors contiguous
    original_save = Trainer._save

    def patched_save(self, output_dir, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        for k in state_dict:
            if not state_dict[k].is_contiguous():
                state_dict[k] = state_dict[k].contiguous()
        os.makedirs(output_dir, exist_ok=True)       
        torch.save(state_dict, f"{output_dir}/pytorch_model.bin")

    Trainer._save = patched_save

    # Training args
    training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,  # You can set higher epochs if using early stopping
    learning_rate=2e-5,   # 👈 Lower LR
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_strategy='epoch',
    logging_steps=1,
    report_to='none',
    save_total_limit=2  # Optional: keep only the 2 best checkpoints
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            LogCallback(),  # Optional custom logging from previous answer
        ]
    )

    # Train and evaluate
    trainer.train()
    print("✅ Training complete.")

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print("\n📊 Test Metrics:")
    print(test_results)

    # Save predictions
    preds = trainer.predict(test_dataset)
    pred_labels = np.argmax(preds.predictions, axis=1)
    decoded_preds = encoder.inverse_transform(pred_labels)
    decoded_true = encoder.inverse_transform(y_test)

    pd.DataFrame({
        "description": X_test.reset_index(drop=True),
        "true_label": decoded_true,
        "predicted_label": decoded_preds
    }).to_excel("test_predictions.xlsx", index=False)
    print("📁 Saved predictions to test_predictions1.xlsx")
    
    print("\n📋 Classification Report:")
    print(classification_report(decoded_true, decoded_preds))

    conf_mat = confusion_matrix(decoded_true, decoded_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
def plot_class_weights(class_weights, encoder):
    labels = encoder.classes_
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(labels)), class_weights)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel('Class Weight')
    plt.title('Computed Class Weights')
    plt.tight_layout()
    plt.show()
    
# ===============================
# Run
# ===============================
if __name__ == "__main__":
    train_pipeline()


# In[ ]:





# In[ ]:




