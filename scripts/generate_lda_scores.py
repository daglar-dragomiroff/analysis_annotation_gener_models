# Импортируем необходимые библиотеки
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.rouge_score import rouge_n, rouge_l
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score
from cider_scorer import CiderScorer

# Загрузка данных
def load_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            data.append(text)
    return data

# Вычисление BLEU
def compute_bleu(reference, hypothesis):
    return corpus_bleu([reference], [hypothesis])

# Вычисление ROUGE
def compute_rouge(reference, hypothesis):
    return rouge_l(reference, hypothesis)

# Вычисление METEOR
def compute_meteor(reference, hypothesis):
    return meteor_score(reference, hypothesis)

# Вычисление F1
def compute_f1(reference, hypothesis):
    return f1_score(reference, hypothesis, average='weighted')

# Вычисление CIDEr
def compute_cider(reference, hypothesis):
    cider_scorer = CiderScorer(n=4, sigma=6)
    cider_scorer += (reference, hypothesis)
    return cider_scorer.compute_score()['CIDEr']

# Путь к данным
rnn_folder = 'annotations/RNN'
bert_folder = 'annotations/BERT'

# Загрузка данных
rnn_data = load_data(rnn_folder)
bert_data = load_data(bert_folder)

# Вычисление метрик
bleu_scores = []
rouge_scores = []
meteor_scores = []
f1_scores = []
cider_scores = []

for rnn_text, bert_text in zip(rnn_data, bert_data):
    bleu_scores.append(compute_bleu(rnn_text, bert_text))
    rouge_scores.append(compute_rouge(rnn_text, bert_text))
    meteor_scores.append(compute_meteor(rnn_text, bert_text))
    f1_scores.append(compute_f1(rnn_text, bert_text))
    cider_scores.append(compute_cider(rnn_text, bert_text))

# Вычисление средних значений
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_rouge = sum(rouge_scores) / len(rouge_scores)
avg_meteor = sum(meteor_scores) / len(meteor_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)
avg_cider = sum(cider_scores) / len(cider_scores)

# Запись результатов
with open('LDA Scores/LDA.BLEU.txt', 'w') as file:
    file.write(str(avg_bleu))

with open('LDA Scores/LDA.ROUGE.txt', 'w') as file:
    file.write(str(avg_rouge))

with open('LDA Scores/LDA.METEOR.txt', 'w') as file:
    file.write(str(avg_meteor))

with open('LDA Scores/LDA.F1.txt', 'w') as file:
    file.write(str(avg_f1))

with open('LDA Scores/LDA.CIDEr.txt', 'w') as file:
    file.write(str(avg_cider))

print("Результаты записаны в файлы LDA Scores.")
