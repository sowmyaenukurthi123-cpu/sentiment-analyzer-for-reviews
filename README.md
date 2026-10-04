# Review Sentiment Analyzer (Google Colab)

Classify product, movie, app or restaurant reviews as **Positive**, **Neutral** or **Negative** using a pretrained transformer model. Runs in a single Google Colab cell, with no training or setup needed.

## Features

- Pretrained RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- Three-class output with a confidence score
- Batch analysis of many reviews at once
- Bar and pie charts of the sentiment distribution
- Interactive text box to test your own reviews
- Optional CSV upload with downloadable results

## Quick Start

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook.
2. Copy the contents of `sentiment_analyzer_colab.py` into a cell.
3. Click **Run**. The first run downloads the model, which takes a minute or two.
4. (Optional) Go to **Runtime > Change runtime type > T4 GPU** for faster batch processing.

## Usage

### Analyze a single review

```python
analyze("Great value for the price, would definitely buy again!")
# ('Positive', 0.98)
```

### Analyze a list of reviews

```python
df = analyze_batch(["Loved it!", "Terrible quality.", "It's fine."])
display(df)
plot_summary(df)
```

### Analyze a CSV file

1. Uncomment the last line in the script:
```python
   results = analyze_csv()
```
2. Run the cell and upload your CSV when prompted.
3. Enter the name of the column that contains the review text.
4. The results are shown on screen and saved as `sentiment_results.csv`.

**Input CSV example**

| id | review                       |
|----|------------------------------|
| 1  | Fast delivery, great product |
| 2  | Broke after two days         |

**Output adds two columns:** `sentiment` and `confidence`.

## How It Works

The script loads a Hugging Face `pipeline("sentiment-analysis")` with a RoBERTa model fine-tuned on social media text. Reviews longer than 512 tokens are truncated. Results are collected into a pandas DataFrame and visualized with matplotlib.

## Requirements

Installed automatically by the first line of the script:

- `transformers`
- `torch`
- `pandas`
- `matplotlib`
- `ipywidgets`

## Limitations

- Sarcasm and mixed reviews ("great camera, awful battery") can be misclassified.
- Very long reviews are truncated to 512 tokens.
- The model is trained mainly on English text.
- Confidence scores are not guarantees of correctness.

## Customization

- **Different model:** change the `MODEL` variable. For a binary Positive/Negative model, try `distilbert-base-uncased-finetuned-sst-2-english`.
- **Other languages:** use a multilingual model such as `cardiffnlp/twitter-xlm-roberta-base-sentiment`.
- **Speed:** raise or lower `batch_size` in `analyze_batch()` to suit your GPU memory.

## License

Code: free to use and modify. Model: see the [model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) for its license terms.
