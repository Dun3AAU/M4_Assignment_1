import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def get_attention_data(sentence, embedding_dict):
    words = sentence.lower().split()
    X = np.array([embedding_dict[w].numpy() for w in words])

    #Scaled dot-product attention
    scale = np.sqrt(X.shape[1])
    scores = np.dot(X, X.T) / scale
    weights = softmax(scores)
    output = np.dot(weights, X)
    return words, X, output

def main():
    sentence_1 = "this sample aged like fine wine"
    sentence_2 = "i just got a parking fine"
    
    # 1. Setup Embeddings
    torch.manual_seed(42)
    unique_words = list(set(sentence_1.lower().split() + sentence_2.lower().split()))
    embedding_dict = {word: torch.randn(2) for word in unique_words}

    # 2. Process
    w1, orig1, att1 = get_attention_data(sentence_1, embedding_dict)
    w2, orig2, att2 = get_attention_data(sentence_2, embedding_dict)

    # 3. Calculate Similarity
    idx1, idx2 = w1.index("fine"), w2.index("fine")
    sim_before = cosine_similarity([orig1[idx1]], [orig2[idx2]])[0][0]
    sim_after = cosine_similarity([att1[idx1]], [att2[idx2]])[0][0]
    print(f"Cosine Similarity BEFORE: {sim_before:.4f}")
    print(f"Cosine Similarity AFTER:  {sim_after:.4f}")

    # 4. Plot
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(f"Sent 1: '...fine wine'", f"Sent 2: '...parking fine'"))

    def add_sentence_traces(fig, words, orig, att, col):
        # 1. Original Positions (Gray Circles) - No text to keep it clean
        fig.add_trace(go.Scatter(
            x=orig[:,0], y=orig[:,1],
            mode='markers', # Removed 'text' here to reduce clutter
            marker=dict(size=8, color='gray', opacity=0.3),
            hoverinfo='skip', # Don't show hover for the gray ones
            name=f"Original"
        ), row=1, col=col)

        # 2. Contextual Positions (Diamonds) - HOVER ENABLED
        fig.add_trace(go.Scatter(
            x=att[:,0], y=att[:,1],
            mode='markers',
            text=words, # 
            hovertemplate="<b>%{text}</b><extra></extra>", # Bold text, removes 'trace' name box
            marker=dict(size=12, symbol='diamond', color='blue' if col==1 else 'red'),
            name=f"Contextual"
        ), row=1, col=col)

        # 3. Arrow for 'fine'
        f_idx = words.index("fine")
        fig.add_annotation(
            x=att[f_idx, 0], y=att[f_idx, 1], # New
            ax=orig[f_idx, 0], ay=orig[f_idx, 1], # Old
            xref=f"x{col}", yref=f"y{col}", axref=f"x{col}", ayref=f"y{col}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black"
        )
        
        # 4. Small movement lines for ALL other words (optional, but helpful)
        # This draws faint lines connecting every gray circle to its diamond
        for i in range(len(words)):
            if words[i] == "fine": continue # Skip fine, we have a big arrow
            fig.add_shape(type="line",
                x0=orig[i,0], y0=orig[i,1],
                x1=att[i,0], y1=att[i,1],
                line=dict(color="gray", width=1, dash="dot"),
                opacity=0.3,
                xref=f"x{col}", yref=f"y{col}"
            )

    add_sentence_traces(fig, w1, orig1, att1, 1)
    add_sentence_traces(fig, w2, orig2, att2, 2)

    fig.update_layout(title_text=f"Hover over diamonds to see words (Sim: {sim_before:.2f} -> {sim_after:.2f})", showlegend=False)
    fig.show()

if __name__ == "__main__":
    main()