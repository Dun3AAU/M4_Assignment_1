import numpy as np
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def main():

    sentence_1="This sample aged like fine wine"
    sentence_2="I just got a parking fine"

    words_1 = sentence_1.lower().split()
    words_2 = sentence_2.lower().split()
    unique_words = list(set(words_1 + words_2))

    embedding_dict = {}
    #random embeddings using same seed for reproducibility
    torch.manual_seed(0)
    for word in unique_words:
        embedding_dict[word] = torch.rand(2)

    Q_1 = np.array([embedding_dict[word].numpy() for word in words_1])
    K_1 = np.array([embedding_dict[word].numpy() for word in words_1])
    V_1 = np.array([embedding_dict[word].numpy() for word in words_1])

    attention_scores_1 = np.dot(Q_1, K_1.T)
    attention_weights_1 = np.apply_along_axis(softmax, 1, attention_scores_1)
    attention_output_1 = np.dot(attention_weights_1, V_1)

    df_1 = pd.DataFrame(attention_output_1, columns=['Dim1', 'Dim2'])
    df_1['Word'] = words_1

    sns.scatterplot(data=df_1, x='Dim1', y='Dim2', hue='Word', s=100)
    plt.title("Attention Output Embeddings for Sentence 1")
    plt.show()

    # Sentence 2
    Q_2 = np.array([embedding_dict[word].numpy() for word in words_2])
    K_2 = np.array([embedding_dict[word].numpy() for word in words_2])
    V_2 = np.array([embedding_dict[word].numpy() for word in words_2])

    attention_scores_2 = np.dot(Q_2, K_2.T)
    attention_weights_2 = np.apply_along_axis(softmax, 1, attention_scores_2)
    attention_output_2 = np.dot(attention_weights_2, V_2)

    df_2 = pd.DataFrame(attention_output_2, columns=['Dim1', 'Dim2'])
    df_2['Word'] = words_2

    sns.scatterplot(data=df_2, x='Dim1', y='Dim2', hue='Word', s=100)
    plt.title("Attention Output Embeddings for Sentence 2")
    plt.show()



    fig1 = px.scatter(df_1, x="Dim1", y="Dim2", color="Word",
                    title="Attention Output Embeddings for Sentence 1")
    fig1.show()  # opens in browser

    fig2 = px.scatter(df_2, x="Dim1", y="Dim2", color="Word",
                    title="Attention Output Embeddings for Sentence 2")
    fig2.show()


    #validate with cosine similarity that are dissimilar after attention
    from sklearn.metrics.pairwise import cosine_similarity
    sim_before = cosine_similarity([embedding_dict["fine"].numpy()], [embedding_dict["fine"].numpy()])[0][0]
    sim_after = cosine_similarity([attention_output_2[3]], [attention_output_1[4]])[0][0]
    print(f"Cosine similarity between 'fine' and 'fine' before attention: {sim_before:.4f}")
    print(f"Cosine similarity between 'fine' in both sentences after attention: {sim_after:.4f}")


if __name__ == "__main__":
    main()
