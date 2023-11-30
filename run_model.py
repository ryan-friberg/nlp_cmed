import json
import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding
)
import datasets
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import preprocessing

def run_model(task, model_checkpoint, epochs=3, batch_size=8, lr=1e-5, scheduler="constant_with_warmup", dropout=0.2,
              classification="sequence", chunk_by="sentence", split=False, chunk_size=200):
    """
    Train and evaluate model for given task.

    Arguments:
        task (string): subtask to train and evaluate model on
        model_checkpoint (string): name of BERT pretrained model
        epochs (int): number of training epochs
        batch_size (int): training and evaluation batch size
        lr (float): learning rate
        scheduler (string): Transformers learning rate scheduler to use
        dropout (float): dropout rate
        classification (string): type of classification problem (sequence or token)
        chunk_by (string): how to generate input sequences (sentences or groups of tokens)
        split (boolean): whether to split sequences at medication mentions
        chunk_size (int): number of tokens in each input sequence
    """

    print("Beginning data preprocessing")
    train_out = preprocessing.data_to_json(task, classification, chunk_by, split, chunk_size, train=True)
    dev_out = preprocessing.data_to_json(task, classification, chunk_by, split, chunk_size, train=False)
    print(f"Processed data written to {train_out.name} and {dev_out.name}")
    train_label_counts = get_label_distribution(task, train_out, classification, train=True)
    get_label_distribution(task, dev_out, classification, train=False)

    print("Loading dataset into huggingface format")
    data = get_datasets(task, train_out, dev_out, classification)
    
    print("Tokenizing datasets using BERT")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if classification == "sequence":
        tokenized_data = data.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    elif classification == "token":
        tokenized_data = data.map(lambda x: tokenize_and_align_labels(x, tokenizer, task), batched=True)
    
    print(f"Initializing BERT {classification} classification model")
    int2label = int_to_label(task)
    num_labels = len(int2label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if classification == "sequence":
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                num_labels=num_labels,
                attention_probs_dropout_prob=dropout,
                hidden_dropout_prob=dropout).to(device)
    elif classification == "token":
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
                num_labels=num_labels,
                attention_probs_dropout_prob=dropout,
                hidden_dropout_prob=dropout).to(device)
    print(f"Model sent to {model.device.type}")

    print("Setting up training configuration")
    args = TrainingArguments(
        f"cmed-{task}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=2,
    )

    if classification == "sequence":
        data_collator = DataCollatorWithPadding(tokenizer)
    elif classification == "token":
        data_collator = DataCollatorForTokenClassification(tokenizer)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(tokenized_data["train"])
    num_warmup_steps = num_training_steps // 5
    lr_scheduler = transformers.get_scheduler(
        scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    if classification == "sequence":
        compute_metrics = compute_sequence_metrics
    elif classification == "token":
        compute_metrics = compute_token_metrics

    trainer = CMEDTrainer(
        model=model,
        args=args,
        label_counts=train_label_counts,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["dev"],
        optimizers = (optimizer, lr_scheduler),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    label_names = list(int2label.values())
    print_confusion_matrix(trainer, tokenized_data, classification, label_names)

class CMEDTrainer(Trainer):
    """
    Custom trainer class to implement weighted cross entropy loss for
    unbalanced label distributions.
    """
    def __init__(self, label_counts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_counts = label_counts
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        counts = torch.FloatTensor(self.label_counts).to(model.device)
        weights = 1 - (counts / torch.sum(counts))
        weights = weights / torch.sum(weights)
        loss_func = torch.nn.CrossEntropyLoss(weight=weights)
        
        loss = loss_func(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_token_metrics(p):
    """
    Compute evaluation metrics for given token classification predictions and
    labels.
    
    Arguments:
        p: tuple of model predictions and corresponding gold standard labels
    
    Returns:
        dictionary containing overall accuracy, precision, recall, and f1
        (macro and micro) for medication class
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) and flatten nested lists
    flat_predictions = list()
    flat_labels = list()
    for preds, labels in zip(predictions, labels):
        for p, l in zip(preds, labels):
            if l != -100:
                flat_predictions.append(p)
                flat_labels.append(l)
    
    accuracy = accuracy_score(flat_labels, flat_predictions)
    micro_precision = precision_score(flat_labels, flat_predictions, average='micro')
    macro_precision = precision_score(flat_labels, flat_predictions, average='macro')
    micro_recall = recall_score(flat_labels, flat_predictions, average='micro')
    macro_recall = recall_score(flat_labels, flat_predictions, average='macro')
    micro_f1 = f1_score(flat_labels, flat_predictions, average='micro')
    macro_f1 = f1_score(flat_labels, flat_predictions, average='macro')

    return {
        "accuracy": accuracy,
        "micro_precision": micro_precision,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

def compute_sequence_metrics(p):
    """
    Compute evaluation metrics for given sequence classification predictions and
    labels.
    
    Arguments:
        p: tuple of model predictions and corresponding gold standard labels
    
    Returns:
        dictionary containing overall accuracy, precision (macro and micro),
        recall (macro and micro), and f1 (macro and micro)
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    micro_precision = precision_score(labels, predictions, average='micro')
    macro_precision = precision_score(labels, predictions, average='macro')
    micro_recall = recall_score(labels, predictions, average='micro')
    macro_recall = recall_score(labels, predictions, average='macro')
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')

    return {
        "accuracy": accuracy,
        "micro_precision": micro_precision,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

def print_confusion_matrix(trainer, tokenized_data, classification, label_names):
    """
    Plot confusion matrix.

    Arguments:
        trainer: Transformers trainer object to make predictions
        tokenized_data: dataset containing evaluation data
        classification (string): type of classification being used (sequence or token)
        label_names (list of str): names of labels
    """
    predictions, labels, _ = trainer.predict(tokenized_data["dev"])

    if classification == "sequence":
        predictions = np.argmax(predictions, axis=1)
    elif classification == "token":
        predictions = np.argmax(predictions, axis=2)
        flat_predictions = list()
        flat_labels = list()
        for preds, labels in zip(predictions, labels):
            for p, l in zip(preds, labels):
                if l != -100:
                    flat_predictions.append(p)
                    flat_labels.append(l)
        predictions = flat_predictions
        labels = flat_labels
    
    cm = confusion_matrix(labels, predictions)
    display = ConfusionMatrixDisplay(cm, display_labels=label_names)
    display.plot()
    plt.show()

def get_datasets(task, train_out, dev_out, classification):
    """
    Load dataset for the given task into a Huggingface dataset object.

    Arguments:
        task (string): subtask being performed
        train_out (string): path to training data json file
        dev_out (string): path to dev data json file
        classification (string): type of classification being used (sequence or token)
    
    Returns:
        Huggingface dataset object with training and dev data
    """
    int2label = int_to_label(task)
    num_labels = len(int2label)
    features = {"tokens": datasets.Sequence(datasets.Value("string"))}
    names = list(int2label.values())
    if classification == "sequence":
        features["label"] = datasets.ClassLabel(num_classes=num_labels, names=names)
    elif classification == "token":
        features[f"{task}_tags"] = datasets.Sequence(datasets.ClassLabel(num_classes=num_labels, names=names))
    
    data = load_dataset(
        'json',
        data_files={'train': str(train_out), 'dev': str(dev_out)},
        features=datasets.Features(features)
    )
    return data

def preprocess_function(examples, tokenizer):
    """
    Tokenize inputs with BERT for sequence classification.

    Arguments:
        examples: datapoints to tokenize
        tokenizer: BERT tokenizer to apply to each datapoint
    
    Returns:
        tokenized data
    """
    return tokenizer(examples["tokens"], max_length=512, truncation=True, is_split_into_words=True)

def tokenize_and_align_labels(examples, tokenizer, task):
    """
    Tokenize inputs with BERT and align labels accordingly for token
    classification.

    Arguments:
        examples: datapoints to tokenize
        tokenizer: BERT tokenizer to apply to each datapoint
        task (string): name of task being performed
    """
    tokenized_inputs = tokenizer(examples["tokens"], max_length=512, truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the task.
            else:
                if task == "ner" and label[word_idx] == 1:
                    if label[word_idx] == 1:
                        label_ids.append(2)
                    else:
                        label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
            
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def get_label_distribution(task, out_path, classification, train):
    """
    Calculate label distribution for given task.

    Arguments:
        task (string): subtask being performed
        out_path (Path): path to processed task data
        classification (string): type of classification (sequence or token)
        train (boolean): if distribution is for training set
    """
    int2label = int_to_label(task)
    num_labels = len(int2label)
    counts = [0] * num_labels

    if train:
        print("Train label distribution:")
    else:
        print("Dev label distribution:")

    with out_path.open() as f:
        data = f.readlines()
        for data_string in data:
            data_point = json.loads(data_string)
            if classification == "sequence":
                counts[data_point["label"]] += 1
            elif classification == "token":
                for i in range(num_labels):
                    counts[i] += data_point[f"{task}_tags"].count(i)
    
    for i in range(num_labels):
        print(f"{int2label[i]}: {counts[i]}")
    
    return counts

def int_to_label(task):
    """
    Create dictionary mapping task integer labels to label names.

    Arguments:
        task (string): subtask name
    
    Returns:
        dict mapping integer labels to label names
    """
    if task == "ner":
        classes = ["NoMedication", "MedicationStart", "MedicationContinue"]
    elif task == "event":
        classes = ["NoDisposition", "Disposition", "Undetermined"]
    elif task == "action":
        classes = ["Start", "Stop", "Increase", "Decrease", "UniqueDose"]
    elif task == "temporality":
        classes = ["Past", "Present", "Future"]
    elif task == "certainty":
        classes = ["Certain", "Hypothetical", "Conditional"]
    elif task == "actor":
        classes = ["Physician", "Patient"]
    
    label_dict = {i: label for i, label in enumerate(classes)}
    return label_dict
