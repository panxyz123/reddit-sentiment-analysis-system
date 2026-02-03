
# load environment; check GPU
from install import *       # run only if restart the kernel
install_requirements()
from utils import *
setup_chapter()

# load dataset
from datasets import load_dataset
emotions = load_dataset("emotion")

# download tokenizer
from transformers import AutoTokenizer
model_ckpt = "/mnt/workspace/distilbert-base-uncased"  # replace with your local model path
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, local_files_only=True)

# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# 1. sklearn classifier with transformer hidden states
from transformers import AutoModel
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt,mirror="tuna").to(device)

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract last hidden states
    lhd = outputs.last_hidden_state[:,0]

    # Return vector for [CLS] token
    return {"hidden_state": lhd}

emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])

## Set up characteristics matrix
import numpy as np
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])


## Train a classifier
labels = emotions["train"].features["label"].names

## SGDC model:
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=3000)
sgd_clf.fit(X_train, y_train)
y_preds = sgd_clf.predict(X_valid)
print('acc of SGDClassifier:')
print(sgd_clf.score(X_valid, y_valid))


#2. dummy classifier (silly classification, 35% accuracy)
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
print('acc of dummy classification:')
print(dummy_clf.score(X_valid, y_valid))

# plot confusion matrix (to analyze connection between predictions and labels)
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
# import matplotlib.pyplot as plt
def plot_confusion_matrix(y_preds, y_true, labels, name):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    style.use("plotting.mplstyle")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.tight_layout()
    # plt.savefig(name)
    plt.show()
plot_confusion_matrix(y_preds, y_valid, labels, "Normalized confusion matrix of LR.png")


#3. End-to end Fine tuning transformers
from transformers import AutoModelForSequenceClassification
num_labels = 6     # 6 classes
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

## define evaluation metrics
from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

## train the model
from transformers import Trainer, TrainingArguments

batch_size = 8
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=3,
                                  learning_rate=3e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level="error")

from transformers import Trainer
trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()

## validation;
preds_output = trainer.predict(emotions_encoded["validation"])
print('fine tuning prediction:')
print(preds_output.metrics)
y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, y_valid, labels, "Normalized confusion matrix (fine tuning of LR).png")

## compute loss, analyze errors
from torch.nn.functional import cross_entropy
def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                             reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(),
            "predicted_label": pred_label.cpu().numpy()}

## Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
## Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16)

emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"]
                              .apply(label_int2str))


# inference: predict new texts
from transformers import pipeline
classifier = pipeline("text-classification", model="./model_LR")
labels = ['sadness','joy','love','anger','fear','surprise']

custom_tweet_list = [ 'i once thought id be cool with his marriage.', 'hes annoyingly consistent with his daily texts. but its strange i miss them when he skips a day.', 'he acted so nice when we met... no wonder i trusted him all those years', 'Another policy? Lol. Betting it wont last a week.',"Although annoying sometimes, he iss generally a nise person. ",]
i=1
for custom_tweet in custom_tweet_list:
    preds = classifier(custom_tweet, return_all_scores=True)
    preds_df = pd.DataFrame(preds[0])
    print("predictions:")
    display(preds_df)
    plt.figure(figsize=(8, 6))
    plt.bar(labels, 100 * preds_df["score"], color='C0')
    plt.title(f'"{custom_tweet}"')
    plt.ylabel("Class probability (%)")
    plt.savefig("Custom text_LR_%d.png" % i)
    plt.show()
    i += 1

