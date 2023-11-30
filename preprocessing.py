from pathlib import Path
import json
import spacy

# scispacy object for text processing
disable = ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'ner']
nlp = spacy.load("en_core_sci_sm", disable=disable)
nlp.add_pipe("sentencizer")

def data_to_json(task, classification="sequence", chunk_by="sentence", split=False, chunk_size=200, train=True):
    """
    Write data for given task to a json file that can be loaded into a
    Huggingface data object.

    Arguments:
        data_dir (string): path to directory containing data and annotation files
        task (string): name of task for which to produce data
        spacy_obj: spacy model to use for sentence and word tokenization
        out_path (string): path for json output file
        classification (string): type of classification model (sentence or token)
        chunk_by (string): how to split up data for inputs (sentence or tokens)
        split (bool): whether or not to split chunks by medication (for sentence classification)
        chunk_size (int): size of input chunks if chunking by tokens
    
    Notes:
        - spacy_obj must include sentencizer pipeline to split notes by sentence
        - split argument is ignored in token classification
        - chunk size arugment is ignored in sentence classification
    """
    if task not in ["ner", "event", "action", "temporality", "certainty", "actor"]:
        raise ValueError(f"Unsupported task name {task}")
    
    if classification not in ["sequence", "token"]:
        raise ValueError(f"Unsupported classification type {classification}")
    
    if chunk_by not in ["sentence", "tokens"]:
        raise ValueError(f"Unsupported chunking method {chunk_by}")

    if train:
        data_dir = Path("n2c2Track1TrainingData/data/train")
    else:
        data_dir = Path("n2c2Track1TrainingData/data/dev")
    
    out_path = generate_out_path(task, classification, chunk_by, split, chunk_size, train)
    if out_path.is_file():
        return out_path
    
    data_paths = sorted(list(data_dir.rglob("*.txt")), key=lambda x: x.name)
    ann_paths = sorted(list(data_dir.rglob("*.json")), key=lambda x: x.name)
    assert(len(data_paths) == len(ann_paths))

    data = []
    for data_path, ann_path in zip(data_paths, ann_paths):
        ann_text = ann_path.read_text()
        anns = json.loads(ann_text)
        text = data_path.read_text()
        doc = nlp(text)
        labeled_data = tokenize_and_label(doc, anns, task)
        labeled_chunks = chunkify(doc, labeled_data, chunk_by, chunk_size)
        classification_data = data_for_classification(labeled_chunks, task, classification, split)
        data.extend(classification_data)
    
    with out_path.open(mode="w") as fout:
        for data_point in data:
            fout.write(json.dumps(data_point) + '\n')
    
    return out_path

def tokenize_and_label(doc, anns, task):
    """
    Assign labels to tokenized document.

    Arguments:
        doc: document object produced with spacy
        anns: list of annotation dictionaries for clinical note
        task (string): name of subtask
    
    Returns:
        list of tokens and list of labels for each token
    """
    if task == "certainty":
        task = "certainity"
    
    num_tokens = len(doc)
    if task == "ner":
        labels = [0] * num_tokens
        for ann in anns:
            token_span = ann["token_span"]
            labels[token_span[0]] = 1
            if token_span[1] - token_span[0] > 1:
                start = token_span[0] + 1
                end = token_span[1]
                labels[start:end] = [2] * (end - start)
    else:
        lab2int = label_to_int(task)
        labels = [-100] * num_tokens
        for ann in anns:
            if task in ann:
                token_span = ann["token_span"]
                label = lab2int[ann[task]]
                labels[token_span[0]] = label
    
    tokens = [str(token) for token in doc]
    labels = [ # ignore whitespace tokens
        label if token.strip() != "" else -100 for token, label in zip(tokens, labels)
    ]
    return tokens, labels

def chunkify(doc, data, chunk_by, chunk_size):
    """
    Split data into desired input sequences.

    Arguments:
        doc: document object produced with spacy
        data: tokens and corresponding labels
        chunk_by (string): how to generate input sequences (by sentence or tokens)
        chunk_size (int): if chunking by tokens, size of token chunks
    
    Returns:
        list of dictionaries with tokens and labels for each input sequence
    """
    tokens, labels = data

    chunks = []
    sentence_spans = [(sent.start, sent.end) for sent in doc.sents]
    if chunk_by == "sentence":
        for start, end in sentence_spans:
            data_point = dict()
            data_point["tokens"] = tokens[start:end]
            data_point["labels"] = labels[start:end]
            chunks.append(data_point)
    elif chunk_by == "tokens":
        chunk_span = [0, 0]
        for sent_start, sent_end in sentence_spans:
            if sent_end - chunk_span[0] <= chunk_size:
                chunk_span[1] = sent_end
            else:
                data_point = dict()
                data_point["tokens"] = tokens[slice(*chunk_span)]
                data_point["labels"] = labels[slice(*chunk_span)]
                chunks.append(data_point)
                chunk_span = [sent_start, sent_end]
        data_point = dict()
        data_point["tokens"] = tokens[slice(*chunk_span)]
        data_point["labels"] = labels[slice(*chunk_span)]
        chunks.append(data_point)
    
    return chunks

def data_for_classification(data, task, classification, split):
    """
    Label data according to classification task.

    Arguments:
        data: tokens and labels for each input sequence
        task (string): subtask name
        classification (string): type of classification (sequence or token)
        split (boolean): whether to split sequences at medication mentions
    
    Returns:
        list of dictionaries with tokens and labels for each inputs sequence
    """
    new_data = []
    if classification == "token":
        for chunk in data:
            if len(chunk["labels"]) > chunk["labels"].count(-100):
                data_point = dict()
                data_point["tokens"] = chunk["tokens"].copy()
                data_point[f"{task}_tags"] = chunk["labels"].copy()
                new_data.append(data_point)
    elif classification == "sequence":
        for chunk in data:
            if split:
                labels = [{"label": label, "pos": i} for i, label in enumerate(chunk["labels"]) if label != -100]
                for i, label in enumerate(labels):
                    start = labels[i-1]["pos"] + 1 if i > 0 else 0
                    end = labels[i+1]["pos"] if i < len(labels) - 1 else len(chunk["tokens"])
                    data_point = dict()
                    data_point["tokens"] = chunk["tokens"][start:end]
                    data_point["label"] = label["label"]
                    new_data.append(data_point)
            else:
                labels = [label for label in chunk["labels"] if label != -100]
                for label in labels:
                    data_point = dict()
                    data_point["tokens"] = chunk["tokens"].copy()
                    data_point["label"] = label
                    new_data.append(data_point)
    
    return new_data

def generate_out_path(task, classification, chunk_by, split, chunk_size, train):
    """
    Generate path to write processed data.

    Arguments:
        task (string): subtask name
        classification (string): type of classification (token or sequence)
        chunk_by (string): how to generate input sequences
        split (boolean): whether to split input sequences at medication mention
        chunk_size (int): number of tokens in input sequences
        train (boolean): whether this is training data
    
    Returns:
        Path object
    """
    out_dir = Path("processed_data")
    template = f"{task}_{chunk_by}"
    if chunk_by == "sentence" and split:
        template += "_split"
    elif chunk_by == "tokens":
        template += str(chunk_size)
    
    if train:
        template += "_train"
    else:
        template += "_dev"

    template += f"_for_{classification}_classification.json"
    out_path = out_dir / template
    return out_path

def label_to_int(task):
    """
    Get dictionary mapping label names to integer labels.

    Arguments:
        task (string): subtask name
    
    Returns:
        dictionary mapping label names to integer labels
    """
    if task == "ner":
        classes = ["NoMedication", "MedicationStart", "MedicationContinue"]
    elif task == "event":
        classes = ["NoDisposition", "Disposition", "Undetermined"]
    elif task == "action":
        classes = ["Start", "Stop", "Increase", "Decrease", "UniqueDose"]
    elif task == "temporality":
        classes = ["Past", "Present", "Future"]
    elif task in ["certainty", "certainity"]:
        classes = ["Certain", "Hypothetical", "Conditional"]
    elif task == "actor":
        classes = ["Physician", "Patient"]
    
    label_dict = {label: i for i, label in enumerate(classes)}
    return label_dict
