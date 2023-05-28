from transformers import BertForTokenClassification
def kg_model_entities_generator(model_name,num_labels):
    print('Loading KG BERT model')

    return BertForTokenClassification.from_pretrained(model_name,num_labels=num_labels)