import spacy

def load_engine(engine:str='en_core_web_sm'):
    try:
        nlp = spacy.load(engine)
    except:
        spacy.cli.download(engine)
        nlp = spacy.load(engine)
    return nlp

def func_examples(func, examples):
    print(">> Examples")
    for i, example in enumerate(examples):
        print(f"{i+1}) Before: {example}\n   After: {func(example)}")