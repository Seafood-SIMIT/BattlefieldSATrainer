from transformers import AutoModel
from transformers import BertModel, BertConfig

# 加载预训练模型和分词器
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 定义实体和关系列表
entities = ["敌⽅坦克", "河边", "步兵", "直升机", "我⽅A⾼地", "B盆地", "坦克", "我⽅反坦克武器", "我⽅防空武器"]
relationships = ["前进⾄", "数量", "失守", "回防", "被消灭⾄", "被敌⽅直升机消灭", "⽀援"]

def kg_model_bert_generator(model_name=None,pretrained=None):
    print('Loading KG BERT model')

    return AutoModel.from_pretrained(pretrained)


