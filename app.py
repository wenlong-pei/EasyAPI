# coding=utf8
import string

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import numpy as np
import translators as ts
from transformers import RobertaTokenizer, RobertaForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration,RobertaModel
from openprompt import PromptForClassification
# type_list = ['C#', 'Java', 'JS', 'Python']
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "t5"
pretrainedmodel_path = "E:\python\API-Competion2\codet5-base"  # the path of the pre-trained model
# load plm
from openprompt.plms import load_plm
num_class = 5
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)
# construct hard template
from openprompt.prompts import ManualTemplate

template_text = 'The code {"placeholder":"text_a"} is {"mask"}'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
from openprompt.prompts import ManualVerbalizer

myverbalizer = ManualVerbalizer(tokenizer)
device = torch.device("cpu")
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp / x_sum
    return s.tolist()

# def predict(tokenizer,model,source):
#     encode = tokenizer.encode_plus(source, return_tensors="pt", max_length=512,
#                                         truncation=True, pad_to_max_length=True)
#     source_ids = encode['input_ids'].to(device)
#     source_mask = encode['attention_mask'].to(device)
#     model.eval()
#     result_list = []
#     pred_ids = []
#     num = 0
#     with torch.no_grad():
#         summary_text_ids = model.generate(source_ids,
#                                                attention_mask=source_mask,
#                                                use_cache=True,
#                                                num_beams=10,
#                                                max_length=16,
#                                                num_return_sequences=10)
#         text = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
#                 id in summary_text_ids]
#         num+=1
#         result_list.append(text)
#         if num==5:
#             return result_list
#     return result_list

def predict(tokenizer,model,source):
    # prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    encode = tokenizer.encode_plus(source, return_tensors="pt", max_length=512,
                                         truncation=True, pad_to_max_length=True)
    source_ids = encode['input_ids'].to(device)
    source_mask = encode['attention_mask'].to(device)
    # model.eval()
    # result_list = []
    # pred_ids = []
    # num = 0
    with torch.no_grad():
        outputs = model(input_ids=source_ids,
                        attention_mask=source_mask)['logits']
        logit = outputs[0].numpy().tolist()
        logit = softmax(logit)
        # logits = prompt_model(source)
        # res = torch.argmax(logits, dim=-1).cpu().tolist()
    return logit

def classfication(tokenizer,model,source):
    encode = tokenizer.encode_plus(source, return_tensors="pt", max_length=512,
                                        truncation=True, pad_to_max_length=True)
    source_ids = encode['input_ids'].to(device)
    source_mask = encode['attention_mask'].to(device)
    model.eval()
    result_list = []
    pred_ids = []
    num = 0
    text = tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result_list.append(text)
    return result_list


def assessment(tokenizer,model,source):
    encode = tokenizer.encode_plus(source, return_tensors="pt", max_length=512,
                                        truncation=True, pad_to_max_length=True)
    source_ids = encode['input_ids'].to(device)
    source_mask = encode['attention_mask'].to(device)
    model.eval()
    result_list = []
    pred_ids = []
    num = 0
    text = tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result_list.append(text)
    return result_list

# def get_title(tokenizer, model, prefix, input_text):
#     input_ids = tokenizer(prefix+": "+input_text ,return_tensors="pt", max_length=512, padding="max_length", truncation=True)
#     summary_text_ids = model.generate(
#         input_ids=input_ids["input_ids"],
#         attention_mask=input_ids["attention_mask"],
#         bos_token_id=model.config.bos_token_id,
#         eos_token_id=model.config.eos_token_id,
#         length_penalty=1.2,
#         top_k=5,
#         top_p=0.95,
#         max_length=48,
#         min_length=2,
#         num_beams=3,
#         num_return_sequences=3
#     )
#     result_list = []
#     for i, beam_output in enumerate(summary_text_ids):
#         title = tokenizer.decode(beam_output, skip_special_tokens=True)
#         if (title[-1] in string.punctuation):
#             title = title[:-1] + " " + title[-1]
#         result_list.append(title)
#     return result_list

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['WTF_CSRF_CHECK_DEFAULT'] = False
# from flask.json import JSONEncoder as _JSONEncoder
import json
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        import decimal
        if isinstance(o, decimal.Decimal):
            return float(o)
        super(JSONEncoder, self).default(o)
app.json_encoder = JSONEncoder
CORS(app, supports_credentials=True)


@app.route('/')
def hello():
    return 'hello world!'

@app.route('/smartVA',methods=['GET'])
def so():
    time_start = time.time()
    code = request.values.get('code')
    desc = request.values.get('desc')
    # desc = ts.alibaba(desc, from_language="zh", to_language="en")
    # input_text = ' '.join(code.split()[:256]) + " <code> " + ' '.join(desc.split()[:256])
    # input_text = code + ' <extra_id_0> ' + desc
    input_text = code
    pred_label = predict(pre_tokenizer,pre_model, input_text)
    classification = classfication(class_tokenizer,class_model,input_text)
    ass = assessment(ass_tokenizer,ass_model,input_text)
    # logit,pred_label = predict(class_tokenizer, class_model, input_text)
    # input_text = ' '.join(desc.split()[:256]) + " <code> " + ' '.join(code.split()[:256])
    # title_list = get_title(gen_tokenizer, gen_model, pred_label, input_text)
    # title_list = [ts.alibaba(title, from_language="en", to_language="zh") for title in title_list]
    time_end = time.time()
    # logit_list = []
    # for i in range(len(logit)):
    #     logit_list.append({"value":logit[i],"name":type_list[i]})
    return jsonify({'title_1':pred_label, 'time':round(time_end - time_start,2)})

if __name__ == '__main__':
    pre_tokenizer=RobertaTokenizer.from_pretrained('codet5-base')
    pre_model = T5ForConditionalGeneration.from_pretrained("codet5-base")
    class_tokenizer = RobertaTokenizer.from_pretrained('codet5-base-class')
    class_model = T5ForConditionalGeneration.from_pretrained("codet5-base-class")  # BERT 配置文件
    ass_tokenizer = RobertaTokenizer.from_pretrained('codebert-base-ass')
    ass_model = RobertaModel.from_pretrained("codebert-base-ass")
    # gen_model = T5ForConditionalGeneration.from_pretrained("SOTitle-Gen-T5")
    # gen_tokenizer = T5Tokenizer.from_pretrained("SOTitle-Gen-T5")
    app.run(port=5000)
