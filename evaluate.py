import argparse, re, csv, random
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch
import pandas as pd

parser = argparse.ArgumentParser(description="Инструмент для оценки качества стихотворений на основе базы знаний")

parser.add_argument("--prompts", type=str, required=True, help="Путь к файлу *.csv со списком промптов для генерации стихотворений")
parser.add_argument("--num", type=int, required=True, help="Количество стихотворений для генерации (int)")
parser.add_argument("--lang", type=str, required=True, help="Язык: en/ru")

args = parser.parse_args()

path = args.prompts
num = args.num
lang = args.lang

model_id = "google/gemma-3-1b-it"

# Применяем квантизацию: загружаем модель меньшей размерности
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Инициализация модели из HuggingFace: загружается локально на наше устройство
# Это значит, что она не использует сторонние сервисы, а все вычисления выполняются у нас
model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

# Токенизация тоже производится локально, т.е. на нашем устройстве
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Определяем путь к файлу с промптом на нужном языке
prompt_path ='prompts/meta_prompt.txt' if lang == 'en' else 'prompts/meta_prompt_ru.txt' 

# Открываем файл с нужным промптом
with open(prompt_path) as f:
    system_prompt = f.read()

# Подгрузка базы данных из файла
with open('data/database.json') as f:
    database = f.read()

# Считывание промптов из файла
with open(path, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    prompts = [row for row in csvreader]

eval_results = {"prompt": [], "temperature": [], "result": []}

def collect_outputs(prompt, num):
    # Формирование промпта: системная инструкция из файла склеивается с информацией из базы знаний и текущим промптом для оцеенки системы
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt+database},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt},]
            },
        ],
    ]
    # Формирование входных данных для модели
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs_storage = []

    for _ in range(num):
        eval_results['prompt'].append(prompt["eval_prompts"])
        # Рандомизировать температуру
        temperature = random.uniform(0.4, 1.0)
        eval_results['temperature'].append(temperature)
        # Инференс производится заданное пользователем количество раз итераций
        with torch.inference_mode():
            outputs = model.generate(**inputs, temperature=temperature, top_k=50, max_new_tokens=1024)

        outputs = tokenizer.batch_decode(outputs)
        
        # Добавляем парсинг ответа модели
        pattern = r'<start_of_turn>(.*?)<end_of_turn>'
        matches = re.findall(pattern, outputs[0], re.DOTALL)[1]
        eval_results['result'].append(matches)

for prompt in prompts:
    collect_outputs(prompt, num)
    
df = pd.DataFrame.from_dict(eval_results)
df.to_csv("evaluation/results.csv", index=False)
