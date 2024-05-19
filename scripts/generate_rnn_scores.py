import os
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from cider import Cider

# Загрузим аннотации из RNN и BERT
annotations_path = 'annotations/'
rnn_annotations = []
bert_annotations = []
for filename in os.listdir(annotations_path + 'RNN'):
    with open(os.path.join(annotations_path + 'RNN', filename), 'r', encoding='utf-8') as file:
        rnn_annotations.append(file.read())
    with open(os.path.join(annotations_path + 'BERT', filename), 'r', encoding='utf-8') as file:
        bert_annotations.append(file.read())

# Вычислим BLEU Score
bleu_scores = []
for rnn_ann, bert_ann in zip(rnn_annotations, bert_annotations):
    bleu_scores.append(corpus_bleu([[bert_ann]], [rnn_ann]))

# Вычислим ROUGE Score
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = []
for rnn_ann, bert_ann in zip(rnn_annotations, bert_annotations):
    scores = rouge_scorer.score(rnn_ann, bert_ann)
    rouge_scores.append((scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3)

# Вычислим METEOR Score
meteor_scores = []
for rnn_ann, bert_ann in zip(rnn_annotations, bert_annotations):
    meteor_scores.append(meteor_score([bert_ann], rnn_ann))

# Вычислим F1 Score
f1_scores = []
for rnn_ann, bert_ann in zip(rnn_annotations, bert_annotations):
    f1_scores.append(f1_score([bert_ann], [rnn_ann], average='binary'))

# Вычислим CIDEr
cider_scorer = Cider()
cider_scores = []
for rnn_ann, bert_ann in zip(rnn_annotations, bert_annotations):
    cider_scores.append(cider_scorer.compute_score([bert_ann], [rnn_ann])[0])

# Сохраним результаты
with open('results/evaluations/RNN Scores/RNN.BLEU.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average BLEU Score: {sum(bleu_scores) / len(bleu_scores):.4f}")
with open('results/evaluations/RNN Scores/RNN.ROUGE.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average ROUGE Score: {sum(rouge_scores) / len(rouge_scores):.4f}")
with open('results/evaluations/RNN Scores/RNN.METEOR.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average METEOR Score: {sum(meteor_scores) / len(meteor_scores):.4f}")
with open('results/evaluations/RNN Scores/RNN.F1.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average F1 Score: {sum(f1_scores) / len(f1_scores):.4f}")
with open('results/evaluations/RNN Scores/RNN.CIDEr.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average CIDEr Score: {sum(cider_scores) / len(cider_scores):.4f}")

print("Результаты оценки сохранены.")
