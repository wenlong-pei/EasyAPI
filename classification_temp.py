import torch
import numpy as np

from transformers import RobertaTokenizer, RobertaForSequenceClassification

type_list = ['C#', 'Java', 'JS', 'Python']

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp / x_sum
    return s.tolist()

# 初始化BERT 预训练的模型
tokenizer = RobertaTokenizer.from_pretrained('Title-Classification')
model = RobertaForSequenceClassification.from_pretrained("Title-Classification")# BERT 配置文件

def predict(model, text):
    tokens_pt2 = tokenizer.encode_plus(text,return_tensors="pt", max_length=300, padding="max_length", truncation=True)
    input_ids = tokens_pt2['input_ids']
    attention_mask = tokens_pt2['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)['logits']
        logit = outputs[0].numpy().tolist()
        logit = softmax(logit)
        _, y = torch.max(outputs, dim=1)
        #logit,y.item(),
    return logit, type_list[y.item()]


body = """
    I have a Custom User model that takes user ip address. I want to add the IP address of the user upon completion of the sign up form. Where do I implement the below code? I am not sure whether to put this into my forms.py or views.py file.
    I expect to be able to save the user's ip address into my custom user table upon sign up.
    """

code = """
def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
"""

input_text = ' '.join(code.split()[:256]) + " <code> " + ' '.join(body.split()[:256])
print(predict(model,input_text))

