from pathlib import Path
import csv
import json
import spacy

# paths to data directories
data_path = Path("n2c2Track1TrainingData/data")
train_path = data_path / "train"
dev_path = data_path / "dev"

# paths to data and annotation files
train_data_files = sorted(list(train_path.rglob("*.txt")), key=lambda x: x.name)
train_ann_files = sorted(list(train_path.rglob("*.ann")), key=lambda x: x.name)
dev_data_files = sorted(list(dev_path.rglob("*.txt")), key=lambda x: x.name)
dev_ann_files = sorted(list(dev_path.rglob("*.ann")), key=lambda x: x.name)

# scispacy object for processing
disable = ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'ner']
nlp = spacy.load("en_core_sci_sm", disable=disable)
nlp.add_pipe("sentencizer")

def annotations_to_json(data_path, ann_path, train=True):
    """
    Convert annotation file to json.

    Arguments:
        data_path (Path): path to clinical note text
        ann_path (Path): path to clinical note annotations
        train (boolean): whether this is training data
    """
    text = data_path.read_text()
    doc = nlp(text)

    with ann_path.open() as fann:
        old_anns = list(csv.reader(fann, delimiter="\t"))

    new_anns = []
    for old_ann in old_anns:
        if old_ann[0][0] == "T":
            med_id = old_ann[0]
            col2 = old_ann[1].split()
            start = int(col2[1])
            end = int(col2[-1])
            med_name = old_ann[2]
            new_ann = {
                "med_id": med_id,
                "char_span": (start, end),
                "med_name": med_name
            }
            new_anns.append(new_ann)
    
    for new_ann in new_anns:
        med_id = new_ann["med_id"]
        for old_ann in old_anns:
            if old_ann[0][0] == "E":
                col2 = old_ann[1].split(":")
                if col2[1].strip() == med_id:
                    new_ann["event_id"] = old_ann[0]
                    new_ann["event"] = col2[0]

    for new_ann in new_anns:
        event_id = new_ann["event_id"]
        for old_ann in old_anns:
            if old_ann[0][0] == "A":
                col2 = old_ann[1].split()
                if col2[1] == event_id:
                    context_dim = col2[0].lower()
                    if context_dim == "negation":
                        continue
                    context_label = col2[2]
                    if context_label == "Unknown" or (context_dim == "action" and context_label == "OtherChange"):
                        continue
                    new_ann[context_dim] = context_label
    
    for new_ann in new_anns:
        span = doc.char_span(*new_ann["char_span"], label=new_ann["med_id"], alignment_mode="expand")
        new_ann["token_span"] = (span.start, span.end)
    
    if train:
        dir_path = train_path
    else:
        dir_path = dev_path
    
    out_path = ann_path.with_suffix(".json")
    with out_path.open(mode="w") as fout:
        fout.write(json.dumps(new_anns))

for data_path, ann_path in zip(train_data_files, train_ann_files):
    annotations_to_json(data_path, ann_path, train=True)

for data_path, ann_path in zip(dev_data_files, dev_ann_files):
    annotations_to_json(data_path, ann_path, train=False)