# main.py

import os
import subprocess

# Предобработка данных
subprocess.run(['python', '01_data_preparation.py'])

# Создание моделей из ноутбуков
subprocess.run(['python', '02_model_training_bow.py'])
subprocess.run(['python', '03_model_training_tfidf.py'])
subprocess.run(['python', '04_model_training_lda.py'])
subprocess.run(['python', '05_model_training_rnn.py'])
subprocess.run(['python', '06_fine_tune_bert.py'])

# Оценка моделей
subprocess.run(['python', 'generate_bow_scores.py'])
subprocess.run(['python', 'generate_tfidf_scores.py'])
subprocess.run(['python', 'generate_lda_scores.py'])
subprocess.run(['python', 'generate_rnn_scores.py'])
subprocess.run(['python', 'generate_bert_scores.py'])
