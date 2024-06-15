from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import numpy as np
from opencc import OpenCC
cc = OpenCC('t2s')
import json

device = "cuda" 

model_path = "/home/featurize/work/LLaMA-Factory/saves/qwen2/lora/dpo/checkpoint-280"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate(theme, relevant=""):
    messages = [
        {"role": "system", "content": "你是一个善于写中国古代诗歌的诗人，接下来我会给你提供一个题目和一些相关的诗句，请你围绕这个题目并且从这些诗句中学习写一首古诗。在生成的过程中请你注意中国古代诗歌的格式要求，并且请你表达尽可能多的情感与含义。在最后输出结果时只需要生成诗歌的主体部分。"},
        {"role": "user", "content": f"请你围绕{theme}这个主题写一首古诗, 参考的诗句为{relevant}"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

embedding_model="text-embedding-3-small"
client = OpenAI(base_url="https://api.chatanywhere.tech/v1", api_key="sk-Dxfv0Y6RZbMmsjDsqwadHlWx1BIjMr2J5iq8L1h9hkytBZoB")

def get_embedding(text, model):
  return client.embeddings.create(input=text, model=embedding_model).data[0].embedding

def cos_similarity(target, embedding):
    numerator = np.sum(target * embedding, axis=1)
    denominator = np.sqrt(np.sum(np.square(target)) * np.sum(np.square(embedding),axis=1))
    return numerator / denominator

def index(data_path):
    collected_dataset = []
    with open(data_path, "r") as f:
        data = json.load(f)
    for each in data:
        paragraph = each["paragraphs"]
        for each in paragraph:
            collected_dataset.append(cc.convert(each))
    embedding = [client.embeddings.create(input=i.strip(), model=embedding_model).data[0].embedding for i in collected_dataset]
    return embedding, collected_dataset

embedding_demo, data_tangshi = index("/home/featurize/work/LLaMA-Factory/data/唐诗三百首.json")

def retrieval(target, embedding, dataset):
    search_embedding = np.array(get_embedding(target, model=embedding_model))
    embedding_similarity = cos_similarity(search_embedding,embedding)
    result=[]
    for i in np.argsort(embedding_similarity)[:-6:-1]:
        result.append(dataset[i])
    return result

def combination(target):
    reference = retrieval(target, embedding=embedding_demo, dataset=data_tangshi)
    poem = ""
    for each in reference:
        poem = poem + each
    generation = generate(target, poem)
    return generation

print(combination("美酒"))