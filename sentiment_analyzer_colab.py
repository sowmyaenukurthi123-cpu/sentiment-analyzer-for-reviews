# ============================================================
#  REVIEW SENTIMENT ANALYZER  -  Google Colab (copy into a cell & run)
#  Model: RoBERTa fine-tuned for Positive / Neutral / Negative
#  Tip: Runtime > Change runtime type > T4 GPU for faster batches
# ============================================================

# ---- 1. Install ----
!pip install -q transformers torch pandas matplotlib ipywidgets

# ---- 2. Imports & model ----
import io
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from transformers import pipeline

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = pipeline(
    "sentiment-analysis",
    model=MODEL,
    truncation=True,
    max_length=512,
    device=0 if __import__("torch").cuda.is_available() else -1,
)
print("Model loaded.")

# ---- 3. Core functions ----
def analyze(text):
    """Analyze one review -> (label, confidence)."""
    r = classifier(str(text))[0]
    return r["label"].capitalize(), round(r["score"], 4)

def analyze_batch(reviews, batch_size=32):
    """Analyze a list of reviews -> DataFrame."""
    reviews = [str(r) for r in reviews]
    results = classifier(reviews, batch_size=batch_size)
    return pd.DataFrame({
        "review": reviews,
        "sentiment": [r["label"].capitalize() for r in results],
        "confidence": [round(r["score"], 4) for r in results],
    })

def plot_summary(df):
    """Bar + pie chart of sentiment distribution."""
    counts = df["sentiment"].value_counts()
    colors = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
    c = [colors.get(k, "gray") for k in counts.index]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    counts.plot(kind="bar", color=c, ax=ax[0], rot=0)
    ax[0].set_title("Sentiment Count")
    ax[0].set_ylabel("Reviews")
    ax[1].pie(counts, labels=counts.index, colors=c, autopct="%1.1f%%", startangle=90)
    ax[1].set_title("Sentiment Share")
    plt.tight_layout()
    plt.show()

# ---- 4. Quick demo on sample reviews ----
sample_reviews = [
    "Absolutely love this phone! Battery lasts all day and the camera is superb.",
    "Terrible quality. It broke after two days and customer support ignored me.",
    "It's okay. Does the job, nothing special.",
    "Fast delivery and the packaging was great, but the product felt a bit cheap.",
    "Worst purchase ever. Complete waste of money.",
    "Great value for the price, would definitely buy again!",
]

demo_df = analyze_batch(sample_reviews)
pd.set_option("display.max_colwidth", 80)
display(demo_df)
plot_summary(demo_df)

# ---- 5. Interactive box: type any review ----
box = widgets.Textarea(
    placeholder="Type or paste a review here...",
    layout=widgets.Layout(width="90%", height="90px"),
)
btn = widgets.Button(description="Analyze", button_style="primary", icon="search")
out = widgets.Output()

def on_click(_):
    with out:
        clear_output()
        if not box.value.strip():
            print("Please enter a review.")
            return
        label, score = analyze(box.value)
        emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}.get(label, "")
        print(f"Sentiment : {label} {emoji}\nConfidence: {score:.2%}")

btn.on_click(on_click)
print("\n--- Try your own review ---")
display(box, btn, out)

# ---- 6. Analyze your own CSV file (optional) ----
# Run this part when you want to analyze a file of reviews.
# Your CSV needs a column containing the review text.
def analyze_csv():
    from google.colab import files
    uploaded = files.upload()
    name = next(iter(uploaded))
    df = pd.read_csv(io.BytesIO(uploaded[name]))
    print("Columns found:", list(df.columns))

    col = input("Type the name of the review column: ").strip()
    df = df.dropna(subset=[col]).reset_index(drop=True)

    res = analyze_batch(df[col].tolist())
    final = pd.concat([df, res[["sentiment", "confidence"]]], axis=1)

    display(final.head(10))
    plot_summary(final)

    final.to_csv("sentiment_results.csv", index=False)
    files.download("sentiment_results.csv")
    return final

# Uncomment the next line to upload & analyze a CSV:
# results = analyze_csv()
