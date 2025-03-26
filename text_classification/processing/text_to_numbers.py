from collections import defaultdict
from typing import List
import pandas as pd

def create_keychain(
    nlp_engine,
    pos_labels: List = ['ADJ','ADP','PUNCT','ADV','AUX','SYM','INTJ','CCONJ','X','NOUN','DET','PROPN','NUM','VERB','PART','PRON','SCONJ',],
        ):

    tag_labels = [label for label in nlp_engine.get_pipe('tagger').labels]
    dep_labels = [label for label in nlp_engine.get_pipe('parser').labels]

    return dict(pos_=pos_labels, tag_=tag_labels, dep_=dep_labels,)

def linguistic_features(dataframe: pd.DataFrame, text_column:str, nlp_engine):
    keychain = create_keychain(nlp_engine)
    df = dataframe.copy()
    df = df.reset_index(drop=True)
    master_data = {}
    # Iterate through Keys
    for key in keychain.keys():
        key_data_ = {}
        # Generate Lists for All Key Values
        for attribute in keychain[key]:
            globals()[f'{key}_list'] = []
            key_data_[attribute] = globals()[f'{key}_list']
        # Set Up DefaultDict to Catch Attributes
        for sentence in df[text_column]:
            counter = defaultdict(int)
            doc = nlp_engine(sentence)
            # Gather Attributes from Sentence Tokens
            for token in doc:
                attribute = token.__getattribute__(key) # One of Possible Attributes in Attribute List
                counter[attribute] += 1 # Builds Counter for Indentified Attribute
            # Transfer DefaultDict to List      
            for attribute in keychain[key]:
                if attribute in counter.keys():
                    key_data_[attribute].append(counter[attribute])
                else:
                    key_data_[attribute].append(0)
        master_data.update(key_data_)
    md = pd.DataFrame(master_data)
    if len(df) > 1:
        df = pd.concat([df,md], axis=1)
    else:
        df = df.reset_index(drop=True)
        md = md.reset_index(drop=True)
        df = pd.concat([df,md], axis=1)
    return df