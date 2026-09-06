# PHASE 6 — Sentiment Analysis

**Duration**: 2 Weeks  
**Started**: Week 13  
**Status**: 📝 Pending  
**Goal**: Add sentiment analysis from financial news

---

## 🎯 Objectives

1. Preprocess financial news corpus
2. Implement FinBERT sentiment analyzer
3. Try FinGPT for comparison
4. Build sentiment scoring API
5. Create historical sentiment database

---

## 📰 News Data Pipeline

### **Data Sources**
- **The Daily Star (Business)**: Primary source
- **Dhaka Tribune**: News coverage
- **Reuters Bangladesh**: International news
- **Bloomberg South Asia**: Financial news
- **BD News Today**: Local financial news

### **Scraping Strategy**
```python
import requests
from bs4 import BeautifulSoup
from datetime import datetime

class NewsScraper:
    def __init__(self, sources):
        self.sources = sources
        self.articles = []
    
    def scrape_daily_star(self, start_date, end_date):
        """Scrape articles from The Daily Star Business section"""
        url = "https://www.thedailystar.net/business"
        articles = []
        
        page = 1
        while True:
            page_url = f"{url}?page={page}"
            response = requests.get(page_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            for article in soup.find_all('article'):
                title = article.find('h3').text
                link = article.find('a')['href']
                date_str = article.find('time')['datetime']
                date = datetime.fromisoformat(date_str)
                
                if date < start_date:
                    return articles
                
                # Fetch article content
                article_content = self.fetch_article_content(link)
                
                articles.append({
                    'date': date,
                    'headline': title,
                    'content': article_content,
                    'source': 'Daily Star'
                })
            
            page += 1
    
    def fetch_article_content(self, url):
        """Fetch full article text"""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        return content
```

### **Text Preprocessing**
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_companies(self, text, company_list):
        """Extract company mentions from text"""
        mentioned = []
        text_lower = text.lower()
        for company in company_list:
            if company.lower() in text_lower:
                mentioned.append(company)
        return mentioned
```

---

## 🤖 Model 1: FinBERT

### **Implementation**
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class FinBertSentimentAnalyzer:
    def __init__(self, model_name='ProsusAI/finbert'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = ['positive', 'negative', 'neutral']
    
    def analyze(self, text):
        """Analyze sentiment of text"""
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract scores
        scores = predictions[0].tolist()
        sentiment = self.labels[scores.index(max(scores))]
        confidence = max(scores)
        
        return {
            'sentiment': sentiment,
            'scores': {
                'positive': scores[0],
                'negative': scores[1],
                'neutral': scores[2]
            },
            'confidence': confidence
        }
    
    def analyze_batch(self, texts, batch_size=32):
        """Analyze multiple texts efficiently"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            for j, pred in enumerate(predictions):
                scores = pred.tolist()
                sentiment = self.labels[scores.index(max(scores))]
                results.append({
                    'sentiment': sentiment,
                    'scores': dict(zip(self.labels, scores)),
                    'confidence': max(scores)
                })
        
        return results
```

### **Usage Example**
```python
analyzer = FinBertSentimentAnalyzer()

texts = [
    "Grameenphone reports record profit, stock surges 5%",
    "BEXIMCO faces legal challenges, shares drop 8%",
    "DSEX trading volume remains stable"
]

results = analyzer.analyze_batch(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['sentiment']} ({result['confidence']:.2f})")
    print(f"Scores: {result['scores']}\n")
```

### **Fine-tuning on Bangladesh Data**
```python
from transformers import AdamW
from torch.utils.data import DataLoader

def finetune_finbert(train_dataset, val_dataset, epochs=3):
    """Fine-tune FinBERT on Bangladesh financial news"""
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = tokenizer(
                batch['text'], 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            
            outputs = model(**inputs, labels=batch['label'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
    return model
```

---

## 🤖 Model 2: FinGPT Sentiment

### **Implementation**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class FinGPTSentimentAnalyzer:
    def __init__(self, model_name='FinGPT/fingpt-sentiment_llama2-13b'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
    
    def analyze(self, text):
        """Use FinGPT for sentiment analysis with explanation"""
        prompt = f"""Analyze the sentiment of the following financial news article. 
        Classify it as positive, negative, or neutral, and provide a brief explanation.

        Article: {text}
        
        Sentiment: """
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment, explanation = self.parse_result(result)
        
        return {
            'sentiment': sentiment,
            'explanation': explanation,
            'confidence': 0.85  # FinGPT doesn't provide direct confidence
        }
    
    def parse_result(self, result):
        """Parse FinGPT output"""
        # Extract sentiment and explanation
        sentiment_match = re.search(r'(positive|negative|neutral)', 
                                    result.lower())
        sentiment = sentiment_match.group(1) if sentiment_match else 'neutral'
        
        # Extract explanation (text after sentiment)
        explanation = result.split('Sentiment:')[-1].strip()
        explanation = explanation.split('\n')[0]
        
        return sentiment, explanation
```

---

## 📊 Sentiment Aggregation

### **Company-Level Sentiment**
```python
class CompanySentimentAnalyzer:
    def __init__(self, sentiment_model='finbert'):
        if sentiment_model == 'finbert':
            self.analyzer = FinBertSentimentAnalyzer()
        elif sentiment_model == 'fingpt':
            self.analyzer = FinGPTSentimentAnalyzer()
    
    def get_company_sentiment(self, articles, company_code, 
                             date_range=None):
        """Aggregate sentiment for a specific company"""
        relevant_articles = [
            a for a in articles 
            if company_code in a['companies_mentioned']
        ]
        
        if date_range:
            relevant_articles = [
                a for a in relevant_articles
                if date_range[0] <= a['date'] <= date_range[1]
            ]
        
        if not relevant_articles:
            return None
        
        # Analyze each article
        sentiments = []
        for article in relevant_articles:
            result = self.analyzer.analyze(article['content'])
            result['date'] = article['date']
            sentiments.append(result)
        
        # Aggregate scores
        sentiment_scores = {
            'positive': np.mean([s['scores']['positive'] 
                                for s in sentiments]),
            'negative': np.mean([s['scores']['negative'] 
                                for s in sentiments]),
            'neutral': np.mean([s['scores']['neutral'] 
                              for s in sentiments])
        }
        
        # Compound score (positive - negative)
        compound = sentiment_scores['positive'] - sentiment_scores['negative']
        
        return {
            'company': company_code,
            'num_articles': len(sentiments),
            'sentiment_scores': sentiment_scores,
            'compound_score': compound,
            'trend': self.calculate_sentiment_trend(sentiments),
            'articles': sentiments
        }
    
    def calculate_sentiment_trend(self, sentiments):
        """Calculate sentiment trend over time"""
        if len(sentiments) < 2:
            return 0
        
        # Sort by date
        sorted_sentiments = sorted(sentiments, key=lambda x: x['date'])
        
        # Calculate linear trend
        scores = [s['scores']['positive'] - s['scores']['negative'] 
                  for s in sorted_sentiments]
        x = np.arange(len(scores))
        
        # Simple linear regression
        trend = np.polyfit(x, scores, 1)[0]
        return trend
```

---

## 💾 Sentiment Database Schema

### **PostgreSQL**
```sql
CREATE TABLE news_sentiment (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    company_code VARCHAR(20),
    headline TEXT NOT NULL,
    content TEXT,
    source VARCHAR(100),
    sentiment VARCHAR(20) NOT NULL,  -- positive/negative/neutral
    positive_score DECIMAL(5,4),
    negative_score DECIMAL(5,4),
    neutral_score DECIMAL(5,4),
    confidence DECIMAL(5,4),
    compound_score DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_date (date),
    INDEX idx_company (company_code),
    INDEX idx_sentiment (sentiment)
);

-- Daily aggregated sentiment
CREATE TABLE daily_sentiment (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    company_code VARCHAR(20) NOT NULL,
    avg_positive DECIMAL(5,4),
    avg_negative DECIMAL(5,4),
    avg_neutral DECIMAL(5,4),
    compound_score DECIMAL(5,4),
    article_count INTEGER,
    
    UNIQUE(date, company_code)
);
```

---

## 📈 Sentiment API

### **FastAPI Endpoint**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str
    model: str = 'finbert'

class SentimentResponse(BaseModel):
    sentiment: str
    scores: dict
    confidence: float
    company_mentions: list

@app.post("/api/sentiment/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of given text"""
    try:
        if request.model == 'finbert':
            analyzer = FinBertSentimentAnalyzer()
        else:
            analyzer = FinGPTSentimentAnalyzer()
        
        result = analyzer.analyze(request.text)
        companies = extract_company_mentions(request.text)
        
        return SentimentResponse(
            sentiment=result['sentiment'],
            scores=result['scores'],
            confidence=result['confidence'],
            company_ments=companies
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiment/company/{code}")
async def get_company_sentiment(code: str, days: int = 30):
    """Get sentiment history for a company"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Query database
    sentiments = get_sentiment_from_db(code, start_date, end_date)
    
    return {
        'company': code,
        'period': f'{start_date.date()} to {end_date.date()}',
        'data': sentiments
    }
```

---

## 📂 Project Structure

```
sentiment/
├── scrapers/
│   ├── daily_star_scraper.py
│   ├── dhaka_tribune_scraper.py
│   └── news_aggregator.py
├── models/
│   ├── finbert_analyzer.py
│   └── fingpt_analyzer.py
├── preprocessing/
│   ├── text_cleaner.py
│   └── company_extractor.py
├── aggregation/
│   └── company_sentiment.py
├── api/
│   └── sentiment_api.py
├── database/
│   ├── schema.sql
│   └── load_sentiments.py
└── results/
    ├── sentiment_history/
    └── visualizations/
```

---

## 📊 Sentiment-Price Correlation Analysis

```python
def analyze_sentiment_price_correlation(stock_data, sentiment_data):
    """Check if sentiment correlates with price movements"""
    # Merge data on date
    merged = pd.merge(stock_data, sentiment_data, on=['date', 'company'])
    
    # Calculate correlation
    correlation = merged['compound_score'].corr(merged['daily_return'])
    
    # Granger causality test
    from statsmodels.tsa.stattools import grangercausalitytests
    test_result = grangercausalitytests(
        merged[['daily_return', 'compound_score']], 
        maxlag=5
    )
    
    return {
        'correlation': correlation,
        'granger_test': test_result,
        'p_value': test_result[1][0]['ssr_chi2test'][1]
    }
```

---

## ✅ Success Criteria

- [ ] News scraping pipeline functional
- [ ] FinBERT model deployed
- [ ] FinGPT model tested
- [ ] Company-level sentiment aggregation working
- [ ] Sentiment database created and populated
- [ ] API endpoints operational
- [ ] Historical sentiment database (2010-2025)
- [ ] Sentiment-price correlation analysis done
- [ ] Visualizations generated

---

## 📊 Evaluation Metrics

### **Model Performance**
- **Accuracy**: % correct sentiment classification
- **F1 Score**: Weighted F1 across classes
- **Precision/Recall**: Per-class metrics
- **Confusion Matrix**: Error analysis

### **Test Set**
- Manually labeled 500+ articles
- Three annotators for reliability
- Cohen's kappa for inter-annotator agreement

---

## 🛠️ Tools & Libraries

- **Hugging Face Transformers**: Pre-trained models
- **FinBERT**: ProsusAI/finbert
- **FinGPT**: Financial LLM
- **NLTK/spaCy**: Text processing
- **BeautifulSoup/Scrapy**: Web scraping
- **FastAPI**: REST API
- **PostgreSQL**: Database storage

---

## 💡 Tips

1. **Batch processing** for efficiency
2. **Cache results** to avoid recomputation
3. **Handle null/empty** text gracefully
4. **Validate company** mentions
5. **Track model versions** for reproducibility
6. **Monitor drift** in sentiment over time

---

**Next Phase**: Phase 7 — Multimodal Forecasting

**Last Updated**: 2026-08-13

---

## ✅ IMPLEMENTATION COMPLETE (Aug 2026)

**Status**: ✅ Phase 6 complete; advanced FinGPT/Bangla-BERT deferred to [phase_6xxx_fingpt_banglabert.md](phase_6xxx_fingpt_banglabert.md).

### What was actually built (vs the original plan above)

| Original plan | Actual implementation | Why |
|---|---|---|
| Live scrape 5 sites (Daily Star, Tribune, etc.) | **Curated synthetic dataset** (`src/data_collection/news_curator.py`, 1,560 labelled articles) | Bangladeshi news sites block bots; no clean RSS for DSE newsroom; Bangla content fragmented across portals |
| FinBERT, FinGPT, comparison | **FinBERT only** (FinGPT needs 16GB+ VRAM) | User has no GPU; FinGPT deferred to 6xxx |
| VADER not in original plan | **Added as fast fallback** | Useful for production speed (sub-ms vs 40ms) |
| NLTK + spaCy preprocessing | **Light regex preprocessing only** | Headlines + body are short; full NLP pipeline not needed for sentiment |
| FastAPI scoring API | **Direct module calls** (`src/sentiment/analyzers.py`) | API deferred to Phase 13 (Dashboard) |
| PostgreSQL historical DB | **CSV files** (`news_scored.csv`, `stock_daily_sentiment.csv`) | 1,560 rows fits in pandas; DB deferred to Phase 13 |

### Key results

- **Corpus size**: 1,560 articles (926 English, 634 Bangla) spanning 2010-01-04 → 2026-08-07
- **Coverage**: 30 DSE stocks + DSEX index, all 13 sectors
- **Event types**: earnings, dividend, expansion, scandal, regulatory, macro
- **Label distribution**: 449 positive / 669 negative / 442 neutral
- **FinBERT vs truth (English, n=926)**: Accuracy **91.4%**, Macro-F1 **0.90**, Weighted-F1 **0.91**
- **Inference speed**: ~40ms/article on CPU, full corpus in 62s
- **Sentiment-price lag correlation** (lag 0/1/2/5/10, 30 stocks):
  - Mean lag-0 r = -0.011 (essentially zero, expected for synthetic data)
  - Lag-1 significant stocks (p<0.05): 2/30 (BEXPHARMA, RENATA — pharma sector)
  - Honest finding: no robust lead-lag on synthetic data, framework is correct

### Files created

```
src/
├── data_collection/
│   └── news_curator.py                     # 1,560-article generator
├── sentiment/
│   ├── __init__.py
│   ├── analyzers.py                        # FinBERT + VADER + BanglaLexicon + Auto
│   ├── scoring_pipeline.py                 # per-article + per-stock-daily scoring
│   ├── correlation_analysis.py             # lag 0/1/2/5/10 Pearson r
│   └── visualize.py                        # 7 PNGs + summary_report.txt

data/raw/news/
└── news_curated.csv                        # labelled corpus

results/sentiment/
├── news_scored.csv                         # per-article scores
├── stock_daily_sentiment.csv               # per-(stock,date) aggregation
├── correlation_per_stock.csv               # 30 stocks × 5 lags
├── correlation_summary.txt                 # human-readable
├── correlation_summary.json                # machine-readable
├── summary_report.txt                      # phase summary
└── plots/
    ├── 01_label_distribution.png
    ├── 02_sentiment_by_sector.png
    ├── 03_sentiment_by_event.png
    ├── 04_daily_sentiment_over_time.png
    ├── 05_correlation_by_lag.png
    ├── 06_finbert_vs_truth_confusion.png
    └── 07_next_return_by_sentiment.png
```

### How to reproduce

```bash
python src/data_collection/news_curator.py        # → data/raw/news/news_curated.csv
python src/sentiment/scoring_pipeline.py          # → news_scored.csv + stock_daily_sentiment.csv
python src/sentiment/correlation_analysis.py      # → correlation_*.csv/json/txt
python src/sentiment/visualize.py                 # → 7 PNGs + summary_report.txt
```

Or in one shot: `python scripts/run_pipeline.py --phase 6`
