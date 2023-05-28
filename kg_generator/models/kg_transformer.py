import torch
from transformers import AutoModelForSequenceClassification


# define the entities and relations

entities =['tank','river']
relationships = ['forward', 'number']

# define the generator

def kg_model_transformer_generator(pretrained_config_file):
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_config_file)
    return model
def generate_kg(sentense):
    # fenci
    inputs = tokenizer(sentense, return_tensors='pt')
    # emotional
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    sentiment = 'positive' if logits[0][0] >logits[0][1] else 'negative'

    # generate the kg according to the emotional

    if sentiment == 'positive':
        kg=[]
        for entity  in entities:
            for relation in relationships:
                if entity in sentense and relation in sentense:
                    kg.append((entity,relation,sentense.split(relation)[1].split()[0]))
        return kg
    else:
        return []