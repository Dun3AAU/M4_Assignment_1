import numpy as np
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def main():

    sentence_1="this sample aged like fine wine"
    sentence_2="i just got a parking fine"

    words_1 = sentence_1.lower().split()
    words_2 = sentence_2.lower().split()
    unique_words = list(set(words_1 + words_2))

    embedding_dict = {}
    #random embeddings using same seed for reproducibility
    torch.manual_seed(42)
    for word in unique_words:
        embedding_dict[word] = torch.randn(2)

    Q_1 = np.array([embedding_dict[word].numpy() for word in words_1])
    K_1 = np.array([embedding_dict[word].numpy() for word in words_1])
    V_1 = np.array([embedding_dict[word].numpy() for word in words_1])

    attention_scores_1 = np.dot(Q_1, K_1.T)
    attention_weights_1 = np.apply_along_axis(softmax, 1, attention_scores_1)
    attention_output_1 = np.dot(attention_weights_1, V_1)

    df_1 = pd.DataFrame(attention_output_1, columns=['Dim1', 'Dim2'])
    df_1['Word'] = words_1

    # Sentence 2
    Q_2 = np.array([embedding_dict[word].numpy() for word in words_2])
    K_2 = np.array([embedding_dict[word].numpy() for word in words_2])
    V_2 = np.array([embedding_dict[word].numpy() for word in words_2])

    attention_scores_2 = np.dot(Q_2, K_2.T)/np.sqrt(2)
    attention_weights_2 = np.apply_along_axis(softmax, 1, attention_scores_2)
    attention_output_2 = np.dot(attention_weights_2, V_2)

    df_2 = pd.DataFrame(attention_output_2, columns=['Dim1', 'Dim2'])
    df_2['Word'] = words_2

    #visualize using plotly, adding a red circle around the word "fine"
    #Add dim1 and dim2 values for word "fine" to highlight
    fig1 = px.scatter(df_1, x="Dim1", y="Dim2", color="Word",
                    title=f"Attention Output Embeddings for Sentence 1. For \"fine\" x: {df_1[df_1['Word']=='fine']['Dim1'].values[0]:.4f}, y: {df_1[df_1['Word']=='fine']['Dim2'].values[0]:.4f}")
    fig1.add_shape(type="circle",
        xref="x", yref="y",
        x0=df_1[df_1['Word']=='fine']['Dim1'].values[0]-0.025,
        y0=df_1[df_1['Word']=='fine']['Dim2'].values[0]-0.05,
        x1=df_1[df_1['Word']=='fine']['Dim1'].values[0]+0.025,
        y1=df_1[df_1['Word']=='fine']['Dim2'].values[0]+0.05,
        line_color="Red",
    )

    fig1.show()  # opens in browser

    fig2 = px.scatter(df_2, x="Dim1", y="Dim2", color="Word",
                    title=f"Attention Output Embeddings for Sentence 2. For \"fine\" x: {df_2[df_2['Word']=='fine']['Dim1'].values[0]:.4f}, y: {df_2[df_2['Word']=='fine']['Dim2'].values[0]:.4f}")
    fig2.add_shape(type="circle",
        xref="x", yref="y",
        x0=df_2[df_2['Word']=='fine']['Dim1'].values[0]-0.025,
        y0=df_2[df_2['Word']=='fine']['Dim2'].values[0]-0.05,
        x1=df_2[df_2['Word']=='fine']['Dim1'].values[0]+0.025,
        y1=df_2[df_2['Word']=='fine']['Dim2'].values[0]+0.05,
        line_color="Red",
    )
    fig2.show()  # opens in browser


    #validate with cosine similarity that fine are dissimilar after attention
    
    sim_before = cosine_similarity([embedding_dict["fine"].numpy()], [embedding_dict["fine"].numpy()])[0][0]
    sim_after = cosine_similarity([attention_output_2[5]], [attention_output_1[4]])[0][0]
    print(f"Cosine similarity between 'fine' and 'fine' before attention: {sim_before:.4f}")
    print(f"Cosine similarity between 'fine' in both sentences after attention: {sim_after:.4f}")


if __name__ == "__main__":
    main()
