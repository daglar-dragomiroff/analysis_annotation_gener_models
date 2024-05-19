import os
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from cider import Cider

# Загрузим аннотации из LDA и BERT
annotations_path = 'annotations/'
lda_annotations = []
bert_annotations = []
for filename in os.listdir(annotations_path + 'LDA'):
    with open(os.path.join(annotations_path + 'LDA', filename), 'r', encoding='utf-8') as file:
        lda_annotations.append(file.read())
    with open(os.path.join(annotations_path + 'BERT', filename), 'r', encoding='utf-8') as file:
        bert_annotations.append(file.read())

# Вычислим BLEU Score
bleu_scores = []
for lda_ann, bert_ann in zip(lda_annotations, bert_annotations):
    bleu_scores.append(corpus_bleu([[bert_ann]], [lda_ann]))

# Вычислим ROUGE Score
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = []
for lda_ann, bert_ann in zip(lda_annotations, bert_annotations):
    scores = rouge_scorer.score(lda_ann, bert_ann)
    rouge_scores.append((scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3)

# Вычислим METEOR Score
meteor_scores = []
for lda_ann, bert_ann in zip(lda_annotations, bert_annotations):
    meteor_scores.append(meteor_score([bert_ann], lda_ann))

# Вычислим F1 Score
f1_scores = []
for lda_ann, bert_ann in zip(lda_annotations, bert_annotations):
    f1_scores.append(f1_score([bert_ann], [lda_ann], average='binary'))

# Вычислим CIDEr
cider_scorer = Cider()
cider_scores = []
for lda_ann, bert_ann in zip(lda_annotations, bert_annotations):
    cider_scores.append(cider_scorer.compute_score([bert_ann], [lda_ann])[0])

# Сохраним результаты
with open('results/evaluations/LDA Scores/LDA.BLEU.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average BLEU Score: {sum(bleu_scores) / len(bleu_scores):.4f}")
with open('results/evaluations/LDA Scores/LDA.ROUGE.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average ROUGE Score: {sum(rouge_scores) / len(rouge_scores):.4f}")
with open('results/evaluations/LDA Scores/LDA.METEOR.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average METEOR Score: {sum(meteor_scores) / len(meteor_scores):.4f}")
with open('results/evaluations/LDA Scores/LDA.F1.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average F1 Score: {sum(f1_scores) / len(f1_scores):.4f}")
with open('results/evaluations/LDA Scores/LDA.CIDEr.txt', 'w', encoding='utf-8') as file:
    file.write(f"Average CIDEr Score: {sum(cider_scores) / len(cider_scores):.4f}")

print("Результаты оценки сохранены.")
