# ML Repository
<br>

ML job scheduler delivering both scheduling via cron jobs and API endpoints for running ETL, training, prediction, evaluation, etc, of various end-to-end models. Developed with FastAPI, TensorFlow, PyTorch, AWS, MongoDB.
<br>
<br>

<figure>
  <img width="944" alt="image" src="https://github.com/user-attachments/assets/80cf4ed7-58f7-4576-a3cd-d63ef584e0fd">
</figure>
Figure 1: High level view of the ML job scheduler and usage in other applications
<br>
<br>

### The ML job scheduler consists of the following models:

1. __BTC forecast__: Bitcoin forecasting
2. __S&P 500__: S&P 500 forecasting
3. __Food classifier__: Food detection / classification
4. __Text analyzer__: NLP text analysis
5. __Text summarizer__: NLP text summarization

### Running the jobs / models:

The jobs / models can be both manually triggered or scheduled via API calls or cron jobs respectively:

1. __BTC forecast__: <em>btc_forecast.py</em>
2. __S&P 500__: <em>sp500_forecast.py</em>
3. __Food classifier__: <em>food_classifier.py</em>
4. __Text analyzer__: <em>text_analyzer.py</em>
5. __Text summarizer__: <em>btc_summarizer.py</em>
