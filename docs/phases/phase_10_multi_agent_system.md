# PHASE 10 — Multi-Agent System

**Duration**: 2 Weeks  
**Started**: Week 20  
**Status**: 📝 Pending  
**Goal**: Build intelligent agent system

---

## 🎯 Objectives

1. Implement 6 specialized agents
2. Build orchestrator for agent coordination
3. Define communication protocol
4. Create agent tools and capabilities
5. Test end-to-end workflow

---

## 🤖 Agent Architecture

```
                        User Query
                            ↓
                  ┌──────────────────────┐
                  │  Orchestrator Agent  │
                  └──────────┬───────────┘
                             ↓
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Prediction   │  │  News Agent  │  │  RAG Agent   │
│   Agent      │  │              │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       ↓                 ↓                  ↓
   Forecast         Sentiment            Context
       │                 │                  │
       └─────────────────┼──────────────────┘
                         ↓
              ┌──────────────────────┐
              │   Portfolio Agent    │
              │  (Risk Assessment)   │
              └──────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │   Advisor Agent      │
              │ (Final Recommendation)│
              └──────────┬───────────┘
                         ↓
                  Final Advice
```

---

## 🤖 Agent 1: Prediction Agent

```python
class PredictionAgent:
    """Forecasts stock prices using trained models"""
    
    def __init__(self, model_registry):
        self.models = model_registry  # Dict of loaded models
        self.feature_pipeline = FeaturePipeline()
    
    def predict(self, stock_code, horizon=5):
        """Predict future returns for a stock"""
        # Load latest data
        data = self.load_latest_data(stock_code)
        
        # Feature engineering
        features = self.feature_pipeline.transform(data)
        
        # Get predictions from multiple models (ensemble)
        predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict(features)
            predictions[model_name] = pred
        
        # Ensemble (weighted average)
        ensemble_pred = self.ensemble_predictions(predictions)
        
        # Calculate confidence intervals
        confidence_interval = self.calculate_confidence(predictions)
        
        # Risk metrics
        risk_metrics = self.calculate_risk_metrics(data, ensemble_pred)
        
        return {
            'stock_code': stock_code,
            'horizon': horizon,
            'forecast_return': ensemble_pred,
            'confidence_interval': confidence_interval,
            'individual_predictions': predictions,
            'risk_metrics': risk_metrics,
            'timestamp': datetime.now()
        }
    
    def ensemble_predictions(self, predictions):
        """Weighted ensemble of multiple models"""
        weights = {
            'transformer': 0.30,
            'informer': 0.25,
            'patchtst': 0.25,
            'xgboost': 0.15,
            'lstm': 0.05
        }
        
        ensemble = sum(
            predictions[model] * weight 
            for model, weight in weights.items()
            if model in predictions
        )
        return ensemble
    
    def calculate_confidence(self, predictions):
        """Calculate confidence based on model agreement"""
        pred_values = list(predictions.values())
        std = np.std(pred_values)
        mean_pred = np.mean(pred_values)
        
        # Lower std = higher confidence
        cv = std / (abs(mean_pred) + 1e-8)
        confidence = max(0, 1 - cv)
        
        return {
            'lower': mean_pred - 1.96 * std,
            'upper': mean_pred + 1.96 * std,
            'confidence': confidence
        }
```

**Tools**:
- `predict(stock_code, horizon)`: Get price forecast
- `get_model_performance(stock_code)`: Model metrics
- `compare_models(stock_code)`: Compare predictions

---

## 🤖 Agent 2: News Agent

```python
class NewsAgent:
    """Analyzes financial news and sentiment"""
    
    def __init__(self, sentiment_analyzer, news_db):
        self.analyzer = sentiment_analyzer  # FinBERT/FinGPT
        self.news_db = news_db
    
    def analyze(self, stock_code, days=7):
        """Analyze recent news for a stock"""
        # Fetch recent news
        articles = self.fetch_news(stock_code, days)
        
        # Sentiment analysis
        sentiments = []
        for article in articles:
            result = self.analyzer.analyze(article['content'])
            result['date'] = article['date']
            result['headline'] = article['headline']
            sentiments.append(result)
        
        # Aggregate
        aggregate = self.aggregate_sentiment(sentiments)
        
        # Identify key themes
        themes = self.extract_themes(articles)
        
        # Impact assessment
        impact = self.assess_impact(aggregate, themes)
        
        return {
            'stock_code': stock_code,
            'period_days': days,
            'num_articles': len(articles),
            'aggregate_sentiment': aggregate,
            'key_themes': themes,
            'market_impact': impact,
            'top_articles': self.get_top_articles(sentiments),
            'timestamp': datetime.now()
        }
    
    def aggregate_sentiment(self, sentiments):
        """Aggregate sentiment scores"""
        return {
            'compound_score': np.mean([
                s['scores']['positive'] - s['scores']['negative'] 
                for s in sentiments
            ]),
            'positive_ratio': np.mean([
                s['scores']['positive'] for s in sentiments
            ]),
            'negative_ratio': np.mean([
                s['scores']['negative'] for s in sentiments
            ]),
            'volatility': np.std([
                s['scores']['positive'] - s['scores']['negative'] 
                for s in sentiments
            ])
        }
    
    def extract_themes(self, articles):
        """Extract key themes using topic modeling"""
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = [a['content'] for a in articles]
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(n_topics=5)
        lda.fit(tfidf)
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [
                vectorizer.get_feature_names()[i] 
                for i in topic.argsort()[-5:]
            ]
            topics.append({
                'topic_id': topic_idx,
                'keywords': top_words
            })
        
        return topics
```

**Tools**:
- `analyze_news(stock_code, days)`: Get news analysis
- `get_sentiment_history(stock_code)`: Historical sentiment
- `detect_events(stock_code)`: Detect major events

---

## 🤖 Agent 3: RAG Agent

```python
class RAGAgent:
    """Retrieves company information from documents"""
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.vectorstore = rag_pipeline.vectorstore
    
    def retrieve_context(self, stock_code, query):
        """Retrieve relevant information from annual reports"""
        # Filter by company
        metadata_filter = {'company': stock_code}
        
        # Retrieve relevant documents
        docs = self.vectorstore.search_by_metadata(
            query, 
            metadata_filter=metadata_filter,
            k=5
        )
        
        # Generate summary
        context = self.synthesize_context(docs, query)
        
        return {
            'stock_code': stock_code,
            'query': query,
            'context': context,
            'source_documents': [
                {
                    'source': doc.metadata.get('source'),
                    'page': doc.metadata.get('page'),
                    'excerpt': doc.page_content[:200]
                }
                for doc in docs
            ],
            'timestamp': datetime.now()
        }
    
    def synthesize_context(self, documents, query):
        """Use LLM to synthesize context"""
        context_text = "\n\n".join([
            doc.page_content for doc in documents
        ])
        
        prompt = f"""Based on the following company documents, 
        answer this question: {query}
        
        Documents:
        {context_text}
        
        Answer:"""
        
        response = self.rag.llm.invoke(prompt)
        return response.content
    
    def get_financial_metrics(self, stock_code):
        """Extract key financial metrics from reports"""
        query = f"What are the key financial metrics for {stock_code}? Include revenue, profit, EPS, and growth rates."
        return self.retrieve_context(stock_code, query)
```

**Tools**:
- `retrieve_context(stock_code, query)`: Get relevant info
- `get_financial_metrics(stock_code)`: Financial data
- `search_documents(query)`: Full-text search

---

## 🤖 Agent 4: Risk Agent

```python
class RiskAgent:
    """Assesses investment risk profiles"""
    
    def __init__(self):
        self.risk_profiles = {
            'conservative': {
                'max_volatility': 0.15,
                'max_drawdown': 0.20,
                'preferred_sectors': ['Telecom', 'Consumer', 'Bank'],
                'allocation': {'stocks': 0.30, 'bonds': 0.60, 'cash': 0.10}
            },
            'moderate': {
                'max_volatility': 0.25,
                'max_drawdown': 0.35,
                'preferred_sectors': ['Bank', 'Pharma', 'Power'],
                'allocation': {'stocks': 0.60, 'bonds': 0.30, 'cash': 0.10}
            },
            'aggressive': {
                'max_volatility': 0.40,
                'max_drawdown': 0.50,
                'preferred_sectors': ['Electronics', 'Pharma', 'Conglomerate'],
                'allocation': {'stocks': 0.85, 'bonds': 0.10, 'cash': 0.05}
            }
        }
    
    def assess(self, stock_code, prediction_data, user_profile=None):
        """Assess risk for a stock"""
        # Calculate stock-specific risk metrics
        stock_risk = self.calculate_stock_risk(stock_code, prediction_data)
        
        # User risk assessment
        if user_profile:
            profile_match = self.match_profile(stock_risk, user_profile)
        else:
            profile_match = None
        
        # Risk-adjusted return
        risk_adjusted = self.calculate_risk_adjusted_return(
            prediction_data, stock_risk
        )
        
        # Recommendations
        recommendation = self.generate_recommendation(
            stock_risk, profile_match, risk_adjusted
        )
        
        return {
            'stock_code': stock_code,
            'stock_risk': stock_risk,
            'user_profile_match': profile_match,
            'risk_adjusted_return': risk_adjusted,
            'recommendation': recommendation,
            'timestamp': datetime.now()
        }
    
    def calculate_stock_risk(self, stock_code, prediction_data):
        """Calculate risk metrics"""
        data = self.load_stock_data(stock_code)
        returns = data['close'].pct_change().dropna()
        
        risk_metrics = {
            'volatility_annual': returns.std() * np.sqrt(252),
            'sharpe_ratio': self.calculate_sharpe(returns),
            'max_drawdown': self.calculate_max_drawdown(data['close']),
            'var_95': returns.quantile(0.05),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            'beta': self.calculate_beta(returns),
            'downside_deviation': returns[returns < 0].std()
        }
        
        return risk_metrics
    
    def match_profile(self, stock_risk, user_profile):
        """Check if stock matches user risk profile"""
        profile_config = self.risk_profiles[user_profile]
        
        checks = {
            'volatility_ok': stock_risk['volatility_annual'] <= profile_config['max_volatility'],
            'drawdown_ok': abs(stock_risk['max_drawdown']) <= profile_config['max_drawdown']
        }
        
        return {
            'profile': user_profile,
            'matches': all(checks.values()),
            'checks': checks
        }
```

**Tools**:
- `assess_risk(stock_code, user_profile)`: Risk assessment
- `get_risk_profile(user_id)`: User profile
- `calculate_var(stock_code, confidence)`: Value at Risk

---

## 🤖 Agent 5: Portfolio Agent

```python
class PortfolioAgent:
    """Optimizes portfolio allocation"""
    
    def __init__(self):
        self.optimizer = PortfolioOptimizer()
    
    def optimize(self, stock_universe, user_profile, investment_amount, 
                constraints=None):
        """Optimize portfolio allocation"""
        # Load expected returns and covariance
        expected_returns = self.get_expected_returns(stock_universe)
        cov_matrix = self.calculate_covariance(stock_universe)
        
        # Apply user constraints
        user_constraints = self.get_user_constraints(
            user_profile, investment_amount
        )
        
        if constraints:
            user_constraints.update(constraints)
        
        # Optimize using Modern Portfolio Theory
        optimization_result = self.optimizer.optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            constraints=user_constraints
        )
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(
            optimization_result['weights'],
            expected_returns,
            cov_matrix
        )
        
        # Generate rebalancing suggestions
        rebalancing = self.suggest_rebalancing(
            optimization_result['weights'],
            user_profile
        )
        
        return {
            'allocation': dict(zip(stock_universe, optimization_result['weights'])),
            'expected_return': portfolio_metrics['expected_return'],
            'volatility': portfolio_metrics['volatility'],
            'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
            'investment_amount': investment_amount,
            'rebalancing_suggestions': rebalancing,
            'efficient_frontier': self.get_efficient_frontier(
                expected_returns, cov_matrix
            ),
            'timestamp': datetime.now()
        }
    
    def calculate_portfolio_metrics(self, weights, returns, cov):
        """Calculate portfolio performance metrics"""
        portfolio_return = np.dot(weights, returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }
```

**Tools**:
- `optimize_portfolio(stocks, profile, amount)`: Optimize allocation
- `get_efficient_frontier(stocks)`: Calculate frontier
- `backtest_portfolio(weights, period)`: Historical backtest

---

## 🤖 Agent 6: Advisor Agent

```python
class AdvisorAgent:
    """Generates final recommendation using LLM reasoning"""
    
    def __init__(self, llm_model='gpt-4'):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.3)
    
    def generate_advice(self, prediction, news, context, risk, portfolio, 
                       user_query):
        """Generate comprehensive investment advice"""
        
        # Aggregate all agent outputs
        analysis = {
            'prediction': prediction,
            'news_sentiment': news,
            'company_context': context,
            'risk_assessment': risk,
            'portfolio_recommendation': portfolio
        }
        
        # Build comprehensive prompt
        prompt = self.build_advisor_prompt(analysis, user_query)
        
        # Generate advice
        response = self.llm.invoke(prompt)
        
        # Parse and structure the advice
        advice = self.parse_advice(response.content)
        
        return {
            'recommendation': advice['recommendation'],
            'confidence': advice['confidence'],
            'reasoning': advice['reasoning'],
            'risks': advice['risks'],
            'time_horizon': advice['time_horizon'],
            'action_items': advice['action_items'],
            'full_response': response.content,
            'underlying_analysis': analysis,
            'timestamp': datetime.now()
        }
    
    def build_advisor_prompt(self, analysis, user_query):
        """Build comprehensive LLM prompt"""
        return f"""You are an expert financial advisor for the Bangladesh 
        stock market. Analyze the following information and provide a clear 
        investment recommendation.

User Question: {user_query}

Analysis Summary:

1. PREDICTION FORECAST:
- Stock: {analysis['prediction']['stock_code']}
- Expected Return: {analysis['prediction']['forecast_return']:.2f}%
- Confidence: {analysis['prediction']['confidence_interval']['confidence']:.2%}
- Risk Level: {analysis['prediction']['risk_metrics']['volatility']:.2%} volatility

2. NEWS SENTIMENT:
- Articles Analyzed: {analysis['news_sentiment']['num_articles']}
- Compound Sentiment: {analysis['news_sentiment']['aggregate_sentiment']['compound_score']:.2f}
- Key Themes: {', '.join([t['keywords'][0] for t in analysis['news_sentiment']['key_themes'][:3]])}

3. COMPANY CONTEXT:
{analysis['company_context']['context'][:500]}

4. RISK ASSESSMENT:
- Sharpe Ratio: {analysis['risk_assessment']['stock_risk']['sharpe_ratio']:.2f}
- Max Drawdown: {analysis['risk_assessment']['stock_risk']['max_drawdown']:.2%}
- Profile Match: {analysis['risk_assessment']['user_profile_match']['matches']}

5. PORTFOLIO RECOMMENDATION:
- Suggested Allocation: {analysis['portfolio_recommendation']['allocation']}
- Expected Portfolio Return: {analysis['portfolio_recommendation']['expected_return']:.2%}
- Sharpe Ratio: {analysis['portfolio_recommendation']['sharpe_ratio']:.2f}

Provide a comprehensive recommendation that includes:
1. Clear BUY/SELL/HOLD decision
2. Confidence level (0-100%)
3. Detailed reasoning (3-5 key points)
4. Risk factors to consider
5. Suggested time horizon
6. Specific action items

Your advice:"""
```

**Tools**:
- `generate_advice(query, context)`: Generate recommendation
- `explain_recommendation(advice_id)`: Explain past advice
- `follow_up_question(advice_id, question)`: Answer follow-ups

---

## 🎼 Orchestrator

```python
class Orchestrator:
    """Coordinates multiple agents to answer user queries"""
    
    def __init__(self):
        self.prediction_agent = PredictionAgent(model_registry)
        self.news_agent = NewsAgent(sentiment_analyzer, news_db)
        self.rag_agent = RAGAgent(rag_pipeline)
        self.risk_agent = RiskAgent()
        self.portfolio_agent = PortfolioAgent()
        self.advisor_agent = AdvisorAgent()
        
        self.query_router = QueryRouter()
    
    def process_query(self, user_query, user_id=None):
        """Main orchestration logic"""
        # Step 1: Parse and route query
        parsed_query = self.query_router.parse(user_query)
        
        # Step 2: Identify required agents
        required_agents = self.query_router.identify_agents(parsed_query)
        
        # Step 3: Execute agents in parallel where possible
        results = {}
        
        # Parallel execution
        if 'prediction' in required_agents:
            results['prediction'] = self.execute_prediction(parsed_query)
        
        if 'news' in required_agents:
            results['news'] = self.execute_news(parsed_query)
        
        if 'rag' in required_agents:
            results['rag'] = self.execute_rag(parsed_query)
        
        # Sequential (depends on previous results)
        if 'risk' in required_agents:
            results['risk'] = self.execute_risk(parsed_query, results)
        
        if 'portfolio' in required_agents:
            results['portfolio'] = self.execute_portfolio(parsed_query, results)
        
        # Final synthesis
        final_advice = self.advisor_agent.generate_advice(
            results.get('prediction'),
            results.get('news'),
            results.get('rag'),
            results.get('risk'),
            results.get('portfolio'),
            user_query
        )
        
        return final_advice
    
    def execute_prediction(self, query):
        """Execute prediction agent"""
        stock_code = query['stock_code']
        return self.prediction_agent.predict(stock_code, horizon=5)
    
    def execute_news(self, query):
        """Execute news agent"""
        stock_code = query['stock_code']
        return self.news_agent.analyze(stock_code, days=7)
    
    def execute_rag(self, query):
        """Execute RAG agent"""
        stock_code = query['stock_code']
        rag_query = query.get('rag_query', f"Tell me about {stock_code}")
        return self.rag_agent.retrieve_context(stock_code, rag_query)
    
    def execute_risk(self, query, previous_results):
        """Execute risk agent"""
        stock_code = query['stock_code']
        user_profile = query.get('user_profile', 'moderate')
        
        return self.risk_agent.assess(
            stock_code,
            previous_results.get('prediction'),
            user_profile
        )
    
    def execute_portfolio(self, query, previous_results):
        """Execute portfolio agent"""
        stock_universe = query.get('stock_universe', [query['stock_code']])
        user_profile = query.get('user_profile', 'moderate')
        amount = query.get('investment_amount', 100000)
        
        return self.portfolio_agent.optimize(
            stock_universe, user_profile, amount
        )
```

---

## 🔀 Query Router

```python
class QueryRouter:
    """Routes queries to appropriate agents"""
    
    def __init__(self):
        self.intent_classifier = self.load_intent_classifier()
        self.entity_extractor = self.load_entity_extractor()
    
    def parse(self, user_query):
        """Parse user query"""
        # Extract intent
        intent = self.classify_intent(user_query)
        
        # Extract entities
        entities = self.extract_entities(user_query)
        
        return {
            'original_query': user_query,
            'intent': intent,
            'stock_code': entities.get('stock_code'),
            'user_profile': entities.get('user_profile', 'moderate'),
            'investment_amount': entities.get('amount', 100000),
            'rag_query': user_query
        }
    
    def classify_intent(self, query):
        """Classify query intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['buy', 'sell', 'should i', 'invest']):
            return 'investment_advice'
        elif any(word in query_lower for word in ['predict', 'forecast', 'will go']):
            return 'prediction'
        elif any(word in query_lower for word in ['risk', 'safe', 'volatile']):
            return 'risk_assessment'
        elif any(word in query_lower for word in ['portfolio', 'allocation', 'diversify']):
            return 'portfolio_optimization'
        elif any(word in query_lower for word in ['news', 'sentiment', 'happening']):
            return 'news_analysis'
        else:
            return 'general_query'
    
    def identify_agents(self, parsed_query):
        """Determine which agents to invoke"""
        intent = parsed_query['intent']
        
        agent_mapping = {
            'investment_advice': ['prediction', 'news', 'rag', 'risk', 'portfolio'],
            'prediction': ['prediction', 'news'],
            'risk_assessment': ['prediction', 'risk'],
            'portfolio_optimization': ['prediction', 'risk', 'portfolio'],
            'news_analysis': ['news', 'rag'],
            'general_query': ['rag']
        }
        
        return agent_mapping.get(intent, ['rag'])
    
    def extract_entities(self, query):
        """Extract stock codes, amounts, etc."""
        # Simple regex-based extraction
        import re
        
        entities = {}
        
        # Extract stock codes (uppercase 2-10 chars)
        stock_match = re.search(r'\b([A-Z]{2,10})\b', query)
        if stock_match:
            entities['stock_code'] = stock_match.group(1)
        
        # Extract amount (numbers followed by BDT, Taka, etc.)
        amount_match = re.search(r'(\d+[,\d]*)\s*(?:BDT|Taka|taka)', query)
        if amount_match:
            entities['amount'] = int(amount_match.group(1).replace(',', ''))
        
        # Extract risk profile
        if 'conservative' in query.lower():
            entities['user_profile'] = 'conservative'
        elif 'aggressive' in query.lower():
            entities['user_profile'] = 'aggressive'
        elif 'moderate' in query.lower():
            entities['user_profile'] = 'moderate'
        
        return entities
```

---

## 🗣️ Communication Protocol

```python
class AgentMessage:
    """Standardized message format for agent communication"""
    
    def __init__(self, sender, receiver, content, message_type='request'):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type  # request, response, broadcast
        self.timestamp = datetime.now()
        self.message_id = str(uuid.uuid4())
    
    def to_dict(self):
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'type': self.message_type,
            'timestamp': self.timestamp.isoformat(),
            'id': self.message_id
        }

class MessageBroker:
    """Manages message passing between agents"""
    
    def __init__(self):
        self.message_queue = Queue()
        self.agent_registry = {}
    
    def send_message(self, message):
        """Send message to receiver"""
        self.message_queue.put(message)
    
    def receive_messages(self, agent_id):
        """Get messages for specific agent"""
        messages = []
        while not self.message_queue.empty():
            msg = self.message_queue.get()
            if msg.receiver == agent_id or msg.receiver == 'broadcast':
                messages.append(msg)
        return messages
```

---

## 📂 Project Structure

```
agents/
├── prediction_agent.py
├── news_agent.py
├── rag_agent.py
├── risk_agent.py
├── portfolio_agent.py
├── advisor_agent.py
├── orchestrator.py
├── query_router.py
├── communication/
│   ├── message.py
│   └── broker.py
├── tools/
│   ├── prediction_tools.py
│   ├── news_tools.py
│   ├── rag_tools.py
│   ├── risk_tools.py
│   └── portfolio_tools.py
└── tests/
    ├── test_prediction_agent.py
    ├── test_orchestrator.py
    └── test_end_to_end.py
```

---

## ✅ Success Criteria

- [ ] All 6 agents implemented and tested
- [ ] Orchestrator working end-to-end
- [ ] Query router correctly classifies intents
- [ ] Agent communication protocol functional
- [ ] Parallel agent execution working
- [ ] Response time < 5 seconds
- [ ] Test cases passing
- [ ] Documentation complete

---

## 🛠️ Tools & Libraries

- **LangChain**: Agent framework
- **AutoGen**: Multi-agent orchestration (alternative)
- **CrewAI**: Agent collaboration
- **OpenAI API**: LLM backbone
- **asyncio**: Async execution

---

## 💡 Best Practices

1. **Clear agent boundaries** - each agent has specific role
2. **Standardized message format** - consistent communication
3. **Parallel execution** where possible - reduce latency
4. **Error handling** - graceful degradation
5. **Caching** - avoid redundant computations
6. **Logging** - track agent decisions
7. **Testing** - unit tests for each agent

---

## 🧪 Example Workflow

```python
# Initialize orchestrator
orchestrator = Orchestrator()

# User query
query = "Should I invest 100,000 BDT in GP stock? I'm a moderate risk investor."

# Process
result = orchestrator.process_query(query, user_id='user_123')

# Output
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['reasoning']}")
print(f"Action Items: {result['action_items']}")
```

**Example Output**:
```
Recommendation: BUY
Confidence: 78%
Reasoning:
1. Strong predicted return of +8.4% over next 5 days
2. Positive news sentiment (compound score: 0.65)
3. Recent earnings beat expectations by 12%
4. RSI indicates oversold conditions (entry point)
5. Matches moderate risk profile (Sharpe: 1.4)

Risks:
- Market volatility currently at 28%
- Sector rotation risk in telecom
- Quarterly results pending

Time Horizon: 3-6 months

Action Items:
1. Allocate 30-40% of investment amount
2. Set stop-loss at -8%
3. Monitor quarterly results (Aug 15)
4. Review position in 2 weeks
```

---

**Next Phase**: Phase 11 — Portfolio Optimization

**Last Updated**: 2026-08-13
