import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df



def build_tfidf(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        max_features=2000
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names, vectorizer


def extract_top_words_per_category(df, vectorizer, feature_names, top_k=60):
    category_word_map = {}

    for category_name in df["category"].unique():
        category_texts = df[df["category"] == category_name]["text"]

        category_matrix = vectorizer.transform(category_texts)
        mean_scores = np.asarray(category_matrix.mean(axis=0)).flatten()

        top_indices = mean_scores.argsort()[-top_k:]
        top_words = [feature_names[i] for i in top_indices]

        category_word_map[category_name] = top_words

    return category_word_map



def load_glove(filepath):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector
    return embeddings


def filter_words_with_glove(category_word_map, glove_embeddings, max_words=40):
    filtered_map = {}

    for category_name, word_list in category_word_map.items():
        valid_words = [w for w in word_list if w in glove_embeddings]
        filtered_map[category_name] = valid_words[:max_words]

    return filtered_map


def prepare_embedding_data(filtered_word_map, glove_embeddings):
    words = []
    vectors = []
    labels = []

    for category_name, word_list in filtered_word_map.items():
        for word in word_list:
            words.append(word)
            vectors.append(glove_embeddings[word])
            labels.append(category_name)

    return words, np.array(vectors), labels



def reduce_dimensions(vectors, method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Only t-SNE implemented")

    reduced = reducer.fit_transform(vectors)
    return reduced



def plot_embeddings_2d(reduced_vectors, labels, words,
                       title, save_path, annotate_n=15):

    plt.figure(figsize=(12, 9))

    unique_categories = sorted(list(set(labels)))

    for category_name in unique_categories:
        x_coords = [
            reduced_vectors[i][0]
            for i in range(len(labels))
            if labels[i] == category_name
        ]

        y_coords = [
            reduced_vectors[i][1]
            for i in range(len(labels))
            if labels[i] == category_name
        ]

        plt.scatter(x_coords, y_coords, label=category_name)

    for i in range(min(annotate_n, len(words))):
        plt.annotate(
            words[i],
            (reduced_vectors[i][0], reduced_vectors[i][1]),
            fontsize=9
        )

    plt.legend()
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def run_glove_embedding_pipeline():

    df = load_dataset("data/bbc_news.csv")
    texts = df["text"].tolist()

    print(f"Loaded {len(texts)} documents")

    tfidf_matrix, feature_names, vectorizer = build_tfidf(texts)

    category_word_map = extract_top_words_per_category(
        df, vectorizer, feature_names
    )

    glove = load_glove("data/glove_50k_50d.txt")
    print(f"Loaded {len(glove)} GloVe vectors")

    filtered_word_map = filter_words_with_glove(category_word_map, glove)

    words, vectors, labels = prepare_embedding_data(
        filtered_word_map, glove
    )

    print(f"Total words: {len(words)}")

    reduced_vectors = reduce_dimensions(vectors, method="tsne")

    plot_embeddings_2d(
        reduced_vectors,
        labels,
        words,
        title="GloVe Word Embeddings (t-SNE)",
        save_path="plots/glove_embedding_plot.png"
    )


if __name__ == "__main__":
    run_glove_embedding_pipeline()