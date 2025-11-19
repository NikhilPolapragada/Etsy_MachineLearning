
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import time
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


train_path = 'etsy_data/data_2025/2025/train'
test_path  = 'etsy_data/data_2025/2025/test'

train_df = pd.read_parquet(train_path)
test_df  = pd.read_parquet(test_path)
print(f"Train shape: {train_df.shape}  |  Test shape: {test_df.shape}")


sns.set(style="whitegrid")

# 1.1 Treemap of Top Categories
counts = train_df['top_category_id'].value_counts().head(20)
labels = [f"{cat}\n{cnt}" for cat, cnt in zip(counts.index, counts.values)]
colors = sns.color_palette("Spectral", len(counts))
plt.figure(figsize=(12, 8))
squarify.plot(sizes=counts.values, label=labels, alpha=0.8, pad=True, color=colors)
legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
legend_labels = [f"{cat}" for cat in counts.index]
legend = plt.legend(handles=legend_handles, labels=legend_labels, 
                    loc='lower left', bbox_to_anchor=(1, 0.5), title="Top Category ID \n ( Highest to Lowest )")

plt.title("Treemap of Top 20 Categories", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

# 1.2 Missing Value Summary Table
missing_table = train_df.isnull().mean().sort_values(ascending=False) * 100
missing_table = missing_table[missing_table > 0].round(2).astype(str) + '%'
missing_df = missing_table.reset_index()
missing_df.columns = ['Column', '% Missing']
plt.figure(figsize=(10, 6))
sns.barplot(data=missing_df, x='% Missing', y='Column', palette='flare')
plt.title("Missing Values by Column", fontsize=14)
plt.xlabel("Percentage Missing", fontsize=12)
plt.ylabel("Column", fontsize=12)
handles = [plt.Rectangle((0,0),1,1, color='orange')]
plt.legend(handles, ["% Missing"], title="Legend", loc='lower right')
plt.tight_layout()
plt.show()
print("\n🔍 Missing Values Summary:")
print(missing_df)

# 1.3 Top Words in Titles (Bar Plot instead of Word Cloud)
text = " ".join(train_df['title'].fillna('').str.lower())
words = text.split()
word_freq = Counter(words)
common_words = pd.DataFrame(word_freq.most_common(15), columns=['Word', 'Count'])
plt.figure(figsize=(12, 7))
barplot = sns.barplot(data=common_words, y='Word', x='Count', palette='mako')
plt.title("Top 15 Frequent Words in Titles", fontsize=14)
plt.xlabel("Word Count", fontsize=12)
plt.ylabel("Word", fontsize=12)
for container in barplot.containers:
    barplot.bar_label(container, fmt='%d', label_type='edge', fontsize=10, padding=3)
handles = [plt.Rectangle((0,0),1,1, color='darkblue')]
plt.legend(handles, ["Word Frequency"], title="Legend", loc='lower right')
plt.tight_layout()
plt.show()


train_df.fillna('', inplace=True)
test_df.fillna('', inplace=True)
le_top = LabelEncoder()
le_bot = LabelEncoder()
train_df['top_enc'] = le_top.fit_transform(train_df['top_category_id'])
train_df['bot_enc'] = le_bot.fit_transform(train_df['bottom_category_id'])
train_df['full_text'] = train_df['title'] + ' ' + train_df['description'] + ' ' + train_df['tags']
test_df['full_text']  = test_df['title']  + ' ' + test_df['description']  + ' ' + test_df['tags']


MAX_WORDS = 10000
MAX_LEN   = 150
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['full_text'])
seq_train = tokenizer.texts_to_sequences(train_df['full_text'])
seq_test  = tokenizer.texts_to_sequences(test_df['full_text'])
X_train = pad_sequences(seq_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_test  = pad_sequences(seq_test,  maxlen=MAX_LEN, padding='post', truncating='post')


X_tr, X_val, y_tr_top, y_val_top, y_tr_bot, y_val_bot = train_test_split(
    X_train, train_df['top_enc'], train_df['bot_enc'], test_size=0.2, random_state=42
)


model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dense(len(le_top.classes_), activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("\nTraining LSTM for Top Category…")
hist = model.fit(X_tr, y_tr_top, validation_data=(X_val, y_val_top),
                 epochs=5, batch_size=64, verbose=1)


for m in ('loss','accuracy'):
    plt.figure(figsize=(10,5))
    plt.plot(hist.history[m], label=f"Train {m.title()}", linewidth=2)
    plt.plot(hist.history[f"val_{m}"], label=f"Validation {m.title()}", linestyle='--', linewidth=2)
    plt.title(f"{m.title()} Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(m.title(), fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


y_pred_top = np.argmax(model.predict(X_val), axis=1)
print("Top Category F1:", f1_score(y_val_top, y_pred_top, average='macro'))
target_names = [str(cls) for cls in le_top.classes_[:15]]
print(classification_report(y_val_top, y_pred_top, target_names=target_names))



print("\n Starting TF-IDF Vectorization for Random Forest…")
tfidf_vec = TfidfVectorizer(max_features=1000)
X_tfidf_train = tfidf_vec.fit_transform(train_df['full_text'])
print(" TF-IDF Vectorization done. Shape:", X_tfidf_train.shape)

print(" Splitting data for Random Forest...")
X_tr_tfidf, X_val_tfidf, y_tr_bot_rf, y_val_bot_rf = train_test_split(
    X_tfidf_train, train_df['bot_enc'], test_size=0.2, random_state=42)
print(" Split done.")

print(" Sampling smaller subset for Random Forest to avoid memory crash...")
sample_size = 30000
X_tr_tfidf_small = X_tr_tfidf[:sample_size]
y_tr_bot_rf_small = y_tr_bot_rf[:sample_size]

print(" Training Random Forest...")
start_time = time.time()
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_tr_tfidf_small, y_tr_bot_rf_small)
print(f" Random Forest trained in {time.time() - start_time:.2f} seconds.")

print(" Predicting Bottom Category...")
y_pred_bot = rf.predict(X_val_tfidf)
print(" Prediction done. Generating report…")

print(" Sample predictions:", y_pred_bot[:10])

present_labels = sorted(np.unique(y_val_bot_rf))
present_names = [str(le_bot.classes_[i]) for i in present_labels]

print(" Classification Report for Bottom Category (Random Forest):")
print(classification_report(y_val_bot_rf, y_pred_bot, labels=present_labels, target_names=present_names))

pd.DataFrame({
    'Actual': le_bot.inverse_transform(y_val_bot_rf),
    'Predicted': le_bot.inverse_transform(y_pred_bot)
}).to_csv('rf_validation_predictions.csv', index=False)
print(" Saved rf_validation_predictions.csv for manual review.")


imps = pd.Series(rf.feature_importances_).sort_values(ascending=False).head(20)
plt.figure(figsize=(10,7))
imps.plot(kind='barh', color=sns.color_palette("magma", 20))
plt.title("Top 20 Random Forest Feature Importances", fontsize=14)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature Index", fontsize=12)
plt.tight_layout()
plt.show()


print("\nRunning PCA + KMeans using TF-IDF vectors for clustering…")
sample_texts = train_df['full_text'].sample(1000, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(sample_texts)

# PCA
X_pca = PCA(n_components=2).fit_transform(X_tfidf.toarray())

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42).fit(X_pca)

# Plot
plt.figure(figsize=(10,8))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=kmeans.labels_,
                palette="Set2", legend='full', s=70, edgecolor="gray")
plt.title("PCA + KMeans Clustering on TF-IDF Vectors", fontsize=14)
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
plt.legend(title="Cluster Label", loc='upper right')
plt.tight_layout()
plt.show()


pred_top    = np.argmax(model.predict(X_test), axis=1)
pred_bottom = rf.predict(tfidf_vec.transform(test_df['full_text']))
submission = pd.DataFrame({
    'product_id': test_df['product_id'],
    'predicted_top_category_id': le_top.inverse_transform(pred_top),
    'predicted_bottom_category_id': le_bot.inverse_transform(pred_bottom)
})
submission.to_csv('submission.csv', index=False)
print(" Saved submission.csv")
