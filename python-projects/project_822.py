# pip install datasets
# pip install numpy
# pip install pandas
# pip install regex 
# pip install textstat 
# pip install spacy
# pip install spacy-arabic
# pip install camel-tools
# pip install transformers
# pip install openpyxl # لتصدير الملف لفتحه في اكسل لكي تظهر الخطوط العربية بشكل افضل من تيرمنال
# pip install scipy
# pip install sentence-transformers
# pip install tensorflow
# pip install tf_keras

# pip install sentence_transformers 

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

# Access the 'by_polishing' subset as an example
by_polishing = dataset["by_polishing"]

# Convert to pandas DataFrame for easier exploration
df = pd.DataFrame(by_polishing)

# Print first 5 entries
print(df.head())

print("\nDatasets info for 'by_polishing':")
print(dataset['by_polishing'])

split = dataset["by_polishing"]

# View one sample
print(split[0])
num_human = len(split["original_abstract"])
# Count human-written abstracts
num_human = len(split["original_abstract"])

# Count AI-generated abstracts (4 per row)
num_ai = len(split["allam_generated_abstract"]) \
       + len(split["jais_generated_abstract"]) \
       + len(split["llama_generated_abstract"]) \
       + len(split["openai_generated_abstract"])

print("Number of human abstracts:", num_human)
print("Number of AI-generated abstracts:", num_ai)

# Distribution ratio
total = num_human + num_ai
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))
dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

import pandas as pd
# Convert to pandas for easier checks
df = pd.DataFrame(split)

# 1. Missing values
print("Missing values per column:")
print(df.isnull().sum())
print("_________________________________________")

# 2. Duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Also check duplicates in each column separately
for col in df.columns:
    print(f"Duplicates in column {col}: {df[col].duplicated().sum()}")
print("_________________________________________")


# 3. Inconsistencies: empty strings or only spaces
for col in df.columns:
    empty_count = df[col].apply(lambda x: str(x).strip() == "").sum()
    print(f"Empty/blank values in column {col}: {empty_count}")

split2 = dataset["from_title"]

# Count human-written abstracts
num_human = len(split2["original_abstract"])

# Count AI-generated abstracts (4 per row)
num_ai = len(split2["allam_generated_abstract"]) \
       + len(split2["jais_generated_abstract"]) \
       + len(split2["llama_generated_abstract"]) \
       + len(split2["openai_generated_abstract"])

print("Number of human abstracts:", num_human)
print("Number of AI-generated abstracts:", num_ai)

# Distribution ratio
total = num_human + num_ai
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))

import pandas as pd
# Convert to pandas for easier checks
df = pd.DataFrame(split2)

# 1. Missing values
print("Missing values per column:")
print(df.isnull().sum())
print("_________________________________________")

# 2. Duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Also check duplicates in each column separately
for col in df.columns:
    print(f"Duplicates in column {col}: {df[col].duplicated().sum()}")
print("_________________________________________")


# 3. Inconsistencies: empty strings or only spaces
for col in df.columns:
    empty_count = df[col].apply(lambda x: str(x).strip() == "").sum()
    print(f"Empty/blank values in column {col}: {empty_count}")

    split3 = dataset["from_title_and_content"]

# Count human-written abstracts
num_human = len(split3["original_abstract"])

# Count AI-generated abstracts (4 per row)
num_ai = len(split3["allam_generated_abstract"]) \
       + len(split3["jais_generated_abstract"]) \
       + len(split3["llama_generated_abstract"]) \
       + len(split3["openai_generated_abstract"])

print("Number of human abstracts:", num_human)
print("Number of AI-generated abstracts:", num_ai)

# Distribution ratio
total = num_human + num_ai
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))

#  2.1: 
#pip install nltk
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from datasets import load_dataset

# Download required NLTK resources
nltk.download('stopwords')
#test features
print(df.head())

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[^؀-ۿ ]+", " ", text)  # remove non-Arabic chars
    return text
    #2.1.1 Normalization


# 2.1.2 aiming to remove altashkeel
def remove_diacritics(text):
    arabic_diacritics = re.compile('[\u0617-\u061A\u064B-\u0652]')
    return re.sub(arabic_diacritics, '', text)

#2.1.3 & 2.1.4
arabic_stopwords = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()

def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [Stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

print(df.columns)

# to overcome a struggle I had to import this library
from nltk.corpus import stopwords
arabic_stopwords = set(stopwords.words("arabic"))

#enhanced the solution via this line
arabic_stopwords = {
    "في", "من", "على", "عن", "إلى", "و", "كما", "أن", "إن", "ما", "هو", "هي","الذي","هذا", "ذلك"
}

def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


text_columns = [
    'original_abstract',
    'allam_generated_abstract',
    'jais_generated_abstract',
    'llama_generated_abstract',
    'openai_generated_abstract'
]

for col in text_columns:
    if col in df.columns:
        clean_col = col + "_clean"
        df[clean_col] = df[col].apply(preprocess_text)
    else:
        print(f"⚠️ Column '{col}' not found in DataFrame!")

print("✅ Preprocessing complete! Here are the new columns:")
print(df.columns)
df.head(2)

text_columns = [
    'original_abstract',
    'allam_generated_abstract',
    'jais_generated_abstract',
    'llama_generated_abstract',
    'openai_generated_abstract'
]
for col in text_columns:
    clean_col = col + "_clean"
    df[clean_col] = df[col].apply(preprocess_text)
print(" Preprocessing complete! Here are the new columns:")
print(df.columns)
df.head(2)

# 2.2
#pip install matpolt lib
#pip install wordcloud
#pip install seaborn
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import numpy as np
#pip install scikit-learn

ai_texts = pd.concat([
    df['allam_generated_abstract_clean'],
    df['jais_generated_abstract_clean'],
    df['llama_generated_abstract_clean'],
    df['openai_generated_abstract_clean']
], axis=0).dropna().tolist()

human_texts = df['original_abstract_clean'].dropna().tolist()

def text_stats(texts):
    words = [w for txt in texts for w in txt.split()]
    avg_word_len = np.mean([len(w) for w in words])
    avg_sent_len = np.mean([len(txt.split()) for txt in texts])
    vocab = set(words)
    ttr = len(vocab) / len(words)
    return avg_word_len, avg_sent_len, ttr

stats_human = text_stats(human_texts)
stats_ai = text_stats(ai_texts)

print("\n Statistical Summary:")
print(f"Human-written: Avg word len={stats_human[0]:.2f}, Avg sent len={stats_human[1]:.2f}, TTR={stats_human[2]:.3f}")
print(f"AI-generated : Avg word len={stats_ai[0]:.2f}, Avg sent len={stats_ai[1]:.2f}, TTR={stats_ai[2]:.3f}")

def plot_top_ngrams(texts, n=2, top_k=15):
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(ngram_range=(n, n))
    bag = vec.fit_transform(texts)
    sum_words = bag.sum(axis=0)
    freqs = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]
    words, counts = zip(*freqs)
    plt.figure(figsize=(10,4))
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f"Top {top_k} {n}-grams – {n}-grams for {'Human' if texts==human_texts else 'AI'} abstracts")
    plt.show()

print("\n🔤 Top Bigrams for Human-written abstracts:")
plot_top_ngrams(human_texts, n=2)

print("\n🔤 Top Bigrams for AI-generated abstracts:")
plot_top_ngrams(ai_texts, n=2)

#here i will try to show tSentence Length Distribution

#Purpose: to Compare the length of the abstracts  (in words or characters).
# bc AI-generated text might be longer, more repetitive, or more uniform than human-written text.

import matplotlib.pyplot as plt

df["human_length"] = df["original_abstract"].apply(lambda x: len(x.split()))
df["ai_length"] = df["openai_generated_abstract"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,5))
plt.hist(df["human_length"], bins=30, alpha=0.6, label="Human-written", color='blue')
plt.hist(df["ai_length"], bins=30, alpha=0.6, label="AI-generated", color='orange')
plt.xlabel("Sentence Length (words)")
plt.ylabel("Frequency")
plt.title("Sentence Length Distribution")
plt.legend()
plt.show()

# TTP=unique words/total words
#to show comaring  lexical diversity between AI-generated vs HUMAN-written vocabulary

def type_token_ratio(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

df["human_ttr"] = df["original_abstract"].apply(type_token_ratio)
df["ai_ttr"] = df["openai_generated_abstract"].apply(type_token_ratio)

plt.figure(figsize=(6,5))
plt.boxplot([df["human_ttr"], df["ai_ttr"]], labels=["Human", "AI"])
plt.title("Vocabulary Richness (Type–Token Ratio)")
plt.ylabel("TTR Score")
plt.show()


#next to show which words are overused by AI vs humans.
from collections import Counter
import pandas as pd

human_words = " ".join(df["original_abstract"]).split()
ai_words = " ".join(df["openai_generated_abstract"]).split()

human_freq = Counter(human_words)
ai_freq = Counter(ai_words)

common_words = set(list(human_freq.keys())[:100]) & set(list(ai_freq.keys())[:100])

data = []
for w in common_words:
    data.append((w, human_freq[w], ai_freq[w]))

freq_df = pd.DataFrame(data, columns=["word", "human", "ai"]).sort_values("human", ascending=False)[:15]

freq_df.plot(x="word", kind="bar", figsize=(10,5), title="Top Words: Human vs AI", rot=45)
plt.ylabel("Frequency")
plt.show()

#area
freq_df.plot(x="word", kind="area", figsize=(10,5), title="Top Words: Human vs AI", rot=45, alpha=0.6)

#line
freq_df.plot(x="word", kind="line", figsize=(10,5), title="Top Words: Human vs AI", rot=45)

#===========================================================================================================
#===========================================================================================================

#TASK 3

#Task 3.1
#important library
import re
import math
import numpy as np
import pandas as pd
import unicodedata
from collections import Counter
from datasets import load_dataset
import regex as re2  # استخدام متقدم (Arabic support)
import string
#import fasttext
import spacy
import torch
import textstat
import os

#imprtant function

def simple_word_tokenize(text):
    """
    تُقسّم النص إلى كلمات ورموز باستخدام دعم العربية (Arabic-aware tokenization)
    """
    return re2.findall(r"\p{Arabic}+|\w+|[^\s\w]", text, flags=re2.VERSION1)

def sentence_tokenize(text):
    """
    تقسيم الجُمل اعتمادًا على علامات الوقف: (., ?, ! , ؛ , ؟)
    """
    parts = re.split(r'(?<=[\.\?\!\u061F\u061B])\s+', text)
    return [p.strip() for p in parts if p.strip()]

original_text_columns = [
    'original_abstract',
    'allam_generated_abstract',
    'jais_generated_abstract',
    'llama_generated_abstract',
    'openai_generated_abstract'
]

clean_text_columns = [
    'original_abstract_clean'
    'allam_generated_abstract_clean',
    'jais_generated_abstract_clean',
    'llama_generated_abstract_clean',
    'openai_generated_abstract_clean'
]

#Tokens, Words, Sentences
# (للأعمدة الأصلية فقط - raw)
# =============================
for col in original_text_columns:
    base = col  # لا حاجة لحذف _clean هنا

    # 3.1 Tokens
    df[f'{base}_tokens'] = df[col].apply(
        lambda t: [tok for tok in simple_word_tokenize(t) if tok.strip()] if isinstance(t, str) else []
    )

    # 4.2 Words (تصفية الرموز غير الحرفية/الرقمية)
    df[f'{base}_words'] = df[f'{base}_tokens'].apply(
        lambda toks: [tok for tok in toks if re.search(r'\w', tok)]
    )

    # 3.3 Sentences
    df[f'{base}_sentences'] = df[col].apply(
        lambda t: sentence_tokenize(t) if isinstance(t, str) else []
    )



def paragraph_tokenize(text):
    """
    تقسيم النص إلى فقرات بناءً على فواصل الأسطر المزدوجة.
    """
    if not isinstance(text, str):
        return []
    # استخدام regex.split للتأكد من التعامل مع الفواصل المختلفة (مثل \n\n أو \r\n\r\n)
    paragraphs = re.split(r'\s*\n\s*\n\s*|\s*\r\n\s*\r\n\s*', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

for col in original_text_columns:
    # إنشاء عمود الفقرات أولاً
    df[f'{col}_paragraphs'] = df[col].apply(paragraph_tokenize)

    #test
print(" تم تجهيز الأعمدة الأساسية والمشتقة (raw only):")
print(df[[c + s for c in original_text_columns for s in ["_tokens","_words","_sentences"]]].head(1).T.head(20))


# دالة مساعدة لحساب عدد علامات POS
def count_pos_tags(pos_tags, target_tag):
    """
    تحسب عدد الكلمات التي تحمل علامة POS محددة في قائمة التاجات.
    """
    # التأكد من أن الإدخال قائمة قبل البدء
    if not isinstance(pos_tags, list):
        return 0
    return sum(1 for word, tag in pos_tags if tag == target_tag)

# Feature 5: Number of elongations 
import re
TATWEEL = '\u0640'
for col in original_text_columns:
    feature = f'{col}_f005_num_elongations'
    def _elong_count(t):
        s = str(t) if pd.notna(t) else ""
        tat = s.count(TATWEEL)
        repeats = len(re.findall(r'(.)\1{2,}', s))
        return tat + repeats
    df[feature] = df[col].apply(_elong_count)


    # Feature 28: Number of colons
# لحساب عدد (:) في الجمل
for col in original_text_columns:
    base = col
    df[f'f028_colons_{base}'] = df[base].apply(
        lambda t: t.count(":") if isinstance(t, str) else 0
    )
    

#df[[f'f028_colons_{col}' for col in original_text_columns]].head()

#51. Number of abbreviations
import re

def count_abbreviations(text):
    """
    Count abbreviations in a text.
    Supports:
    1. English abbreviations (e.g., U.S.A, Ph.D, WHO)
    2. Arabic abbreviations:
       - Letters separated by dots (e.g., د.م، م.ب.د)
       - Short Arabic letter sequences (2-4 letters) within parentheses (e.g., (ص.ب), (م ع))
       - Short Arabic words (2-4 letters) separated by spaces (optional)
    """
    if not isinstance(text, str):
        return 0

    # English abbreviations: U.S.A or WHO
    eng_abbrev_pattern = r'\b(?:[A-Z]\.){2,}|[A-Z]{2,}\b'

    # Arabic abbreviations with dots (e.g., د.م or م.ب.د)
    arabic_with_dots_pattern = r'(?:[\u0621-\u064A]\.){2,}'

    # Arabic short sequences inside parentheses: (ص.ب) أو (م ع)
    arabic_in_parentheses_pattern = r'\([\u0621-\u064A\s\.]{2,10}\)'

    # Optional: Short standalone Arabic words (2-4 letters) as abbreviations
    arabic_short_word_pattern = r'\b[\u0621-\u064A]{2,4}\b'

    # Combine patterns
    combined_pattern = (
        eng_abbrev_pattern + '|' +
        arabic_with_dots_pattern + '|' +
        arabic_in_parentheses_pattern
        # إذا أردت إضافة الكلمات القصيرة كاختصار أضف التالي:
        # + '|' + arabic_short_word_pattern
    )

    matches = re.findall(combined_pattern, text)
    return len(matches)


# التطبيق على الأعمدة الأصلية
for col in original_text_columns:
    df[f'{col}_num_abbreviations'] = df[col].apply(count_abbreviations)



# Feature 74: Number of definitive (ال-) occurrences المعرفات بالألف و اللام
for col in original_text_columns:
    morph_features_col = f'{col}_morph_features'

    if morph_features_col in df.columns:
        # الميزة 74: عدد المعرّفات (Definitives)
        df[f'{col}_f074_num_definitive'] = df[morph_features_col].apply(
            # d.get('is_definite', []) تعيد قائمة بالقيم  (True للمعرّف)
            lambda d: sum(d.get('is_definite', []))
        )

    else:
        # فإذا لم يطبق التحليل الصرفي، نعود إلى قيمة صفر
        df[f'{col}_f074_num_definitive'] = 0



#97. BERT Embedding Similarity
#احسب Mean Cosine Similarity بين embeddings لكل جملة.
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# إعداد الجهاز
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# تحميل موديل LaBSE
model_name = "sentence-transformers/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# دالة للحصول على embedding لكل جملة
def get_sentence_embedding_gpu(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # المتوسط على طول الـ tokens
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # نعيده للـ CPU كـ numpy

# دالة Feature 97
def bert_embedding_similarity(text):
    sentences = sentence_tokenize(text)
    if len(sentences) <= 1:
        return 1.0
    embeddings = np.vstack([get_sentence_embedding_gpu(s) for s in sentences])
    sim_matrix = cosine_similarity(embeddings)
    n = len(sentences)
    sum_sim = np.sum(sim_matrix) - n
    num_pairs = n*(n-1)
    return sum_sim / num_pairs if num_pairs > 0 else 0

# تطبيق الميزة على جميع الأعمدة
for col in original_text_columns:
    print(f"Processing BERT Embedding Similarity for column: {col}")
    df[f'{col}_bert_embedding_similarity'] = df[col].apply(bert_embedding_similarity)

from sklearn.model_selection import train_test_split

# First split: Train 70%, Temp 30%
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)

# Second split: Temp 30% → 15% Validation, 15% Test
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

# Show sizes
print("TOTAL:", len(df))
print("TRAIN:", len(train_df))
print("VAL:", len(val_df))
print("TEST:", len(test_df))

#apply with abstract_text_clean only
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer for Arabic text
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,   # limit vocabulary
    ngram_range=(1,2),   # unigrams + bigrams
    analyzer='word'
)

# Fit only on training set
tfidf_vectorizer.fit(train_df["abstract_text_clean"])

# Transform train/validation/test sets
X_train_tfidf = tfidf_vectorizer.transform(train_df["abstract_text_clean"])
X_val_tfidf   = tfidf_vectorizer.transform(val_df["abstract_text_clean"])
X_test_tfidf  = tfidf_vectorizer.transform(test_df["abstract_text_clean"])

print("TF-IDF shapes:")
print("Train:", X_train_tfidf.shape)
print("Validation:", X_val_tfidf.shape)
print("Test:", X_test_tfidf.shape)

from scipy.sparse import hstack

#Select numeric features (The generated feature engineering exclude label and text)
EXCLUDED_COLS = ['label', 'abstract_text', 'abstract_text_clean',
                 'tokens', 'words', 'sentences', 'paragraphs', 'abstract_text_pos_tags']
# Select columns that are numeric AND not in the exclusion list>>feature engineering columns
numeric_cols = [
    col for col in train_df.select_dtypes(include=np.number).columns.tolist()
    if col not in EXCLUDED_COLS
]
# Convert the numeric features DataFrames to NumPy arrays (dense matrices)
# We must use the values/to_numpy() method to extract the array for sparse matrix stacking.
X_train_num_array = train_df[numeric_cols].values
X_val_num_array   = val_df[numeric_cols].values
X_test_num_array  = test_df[numeric_cols].values


# Target variable
y_train = train_df["label"]
y_val   = val_df["label"]
y_test  = test_df["label"]

# Features: TF-IDF and the creating feature engineering
X_train = hstack([X_train_tfidf, X_train_num_array]).tocsr()
X_val= hstack([X_val_tfidf, X_val_num_array]).tocsr()
X_test= hstack([X_test_tfidf, X_test_num_array]).tocsr()

print("X and y are ready for ML models.")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train on training set
lr_model.fit(X_train, y_train)

# Predict on validation set
y_val_pred = lr_model.predict(X_val)

# Evaluate on validation set
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred))

# Evaluation
# Predict on test set
y_test_pred = lr_model.predict(X_test)

# Evaluate on test set
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))

# Optional: confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Dictionary to store models and results
models = {}

# -----------------------
#Support Vector Machine (SVM)
# -----------------------
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

y_val_pred_svm = svm_model.predict(X_val)
print("SVM Validation Accuracy:", accuracy_score(y_val, y_val_pred_svm))
print(classification_report(y_val, y_val_pred_svm))

models['SVM'] = svm_model

# -----------------------
#Random Forest
# -----------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_val_pred_rf = rf_model.predict(X_val)
print("Random Forest Validation Accuracy:", accuracy_score(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))

models['RandomForest'] = rf_model

# -----------------------
#XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

y_val_pred_xgb = xgb_model.predict(X_val)
print("XGBoost Validation Accuracy:", accuracy_score(y_val, y_val_pred_xgb))
print(classification_report(y_val, y_val_pred_xgb))

models['XGBoost'] = xgb_model

#Evaluation
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# List of models to evaluate
model_names = ['SVM', 'RandomForest', 'XGBoost']

for name in model_names:
    model = models[name]

    # Predict on test set
    y_test_pred = model.predict(X_test)

    print(f"\n===== {name} Test Evaluation =====")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, models['xgb'].predict(X_test)), annot=True, fmt='d')
plt.title('Confusion Matrix (XGBoost)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, models['rf'].predict(X_test)), annot=True, fmt='d')
plt.title('Confusion Matrix (Random Forrest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, models['svm'].predict(X_test)), annot=True, fmt='d')
plt.title('Confusion Matrix (SVM)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, models['lr'].predict(X_test)), annot=True, fmt='d')
plt.title('Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

    from sentence_transformers import SentenceTransformer
import numpy as np

# Load Arabic-compatible BERT model
bert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Convert clean_text into embeddings
X_train_emb = bert_model.encode(train_df["clean_text"].tolist(), convert_to_numpy=True)
X_val_emb   = bert_model.encode(val_df["clean_text"].tolist(), convert_to_numpy=True)
X_test_emb  = bert_model.encode(test_df["clean_text"].tolist(), convert_to_numpy=True)

y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values

print("Train embedding shape:", X_train_emb.shape)

#import tensorflow as tf
from tensorflow.keras import layers, models

# Basic feedforward classifier on embeddings
ffnn_model = models.Sequential([
    layers.Input(shape=(X_train_emb.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")   # binary classification
])

ffnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

ffnn_model.summary()

history = ffnn_model.fit(
    X_train_emb, y_train,
    validation_data=(X_val_emb, y_val),
    epochs=10,
    batch_size=32
)

from sklearn.metrics import accuracy_score, classification_report

# Predict
y_test_pred = (ffnn_model.predict(X_test_emb) > 0.5).astype(int)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

import os
import joblib
from tensorflow.keras.models import Model as KerasModel

def save_all_models(models_dict, save_dir="models"):
    """
    Saves all ML/DL models to disk based on their type.

    Parameters:
    -----------
    models_dict : dict
        Example:
            {
                "logistic_regression": log_reg_model,
                "svm": svm_model,
                "random_forest": rf_model,
                "xgboost": xgb_model,
                "ffnn": ffnn_model
            }

    save_dir : str
        Directory where models will be saved.
    """

    # Create save folder
    os.makedirs(save_dir, exist_ok=True)

    for model_name, model_obj in models_dict.items():

        # Case 1 — Keras deep learning model
        if isinstance(model_obj, KerasModel):
            file_path = os.path.join(save_dir, f"{model_name}.h5")
            model_obj.save(file_path)
            print(f"[Saved] Keras model → {file_path}")

        # Case 2 — All pickle-compatible models (Sklearn, XGBoost)
        else:
            file_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model_obj, file_path)
            print(f"[Saved] Pickle model → {file_path}")

    print("\nAll models saved successfully!")

    import os
os.makedirs("models", exist_ok=True)
models_dict = {
    "lr_model": lr_model,
    "svm": svm_model,
    "random_forest": rf_model,
    "xgboost": xgb_model,
    "ffnn": ffnn_model
}

save_all_models(models_dict)






