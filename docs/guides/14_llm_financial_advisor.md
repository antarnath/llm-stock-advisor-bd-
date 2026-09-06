# PHASE 12 — LLM Financial Advisor

**Duration**: 2 Weeks  
**Started**: Week 23  
**Status**: 📝 Pending  
**Goal**: Integrate all components into cohesive advisor

---

## 🎯 Objectives

1. Build conversational interface
2. Integrate all agents seamlessly
3. Implement context management
4. Add multi-turn dialogue support
5. Create REST API

---

## 🏗️ System Integration

```python
class FinancialAdvisor:
    """Main advisor class integrating all components"""
    
    def __init__(self):
        # Load all components
        self.orchestrator = Orchestrator()
        self.conversation_manager = ConversationManager()
        self.user_profiler = UserProfiler()
        self.explanation_generator = ExplanationGenerator()
        
    def advise(self, user_query, user_id=None):
        """Main entry point for user queries"""
        # Get user context
        user_context = self.user_profiler.get_profile(user_id)
        
        # Process with orchestrator
        result = self.orchestrator.process_query(
            user_query, 
            user_id=user_id
        )
        
        # Generate natural language response
        nl_response = self.generate_response(result, user_context)
        
        # Store in conversation history
        self.conversation_manager.add_message(
            user_id, user_query, nl_response
        )
        
        return nl_response
    
    def generate_response(self, analysis, user_context):
        """Generate comprehensive natural language response"""
        prompt = self.build_response_prompt(analysis, user_context)
        response = self.orchestrator.advisor_agent.llm.invoke(prompt)
        return self.format_response(response.content, analysis)
```

---

## 💬 Conversational Interface

```python
class ConversationManager:
    """Manage multi-turn conversations"""
    
    def __init__(self):
        self.conversations = {}  # user_id -> conversation history
    
    def add_message(self, user_id, query, response):
        """Add message to conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            'timestamp': datetime.now(),
            'query': query,
            'response': response
        })
    
    def get_context(self, user_id, num_messages=5):
        """Get recent conversation context"""
        if user_id not in self.conversations:
            return []
        
        recent = self.conversations[user_id][-num_messages:]
        return recent
    
    def clear_history(self, user_id):
        """Clear conversation for user"""
        if user_id in self.conversations:
            del self.conversations[user_id]
```

### **Context-Aware Responses**

```python
class ContextAwareResponder:
    """Generate context-aware responses using conversation history"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_response(self, current_query, conversation_history, 
                         analysis):
        """Generate response considering conversation history"""
        # Build context
        context_summary = self.summarize_history(conversation_history)
        
        prompt = f"""You are a financial advisor continuing a conversation 
        with a user. Use the previous conversation context and the new 
        analysis to provide a coherent, contextually relevant response.

Previous Conversation:
{context_summary}

Current Query: {current_query}

New Analysis:
{self.format_analysis(analysis)}

Provide a response that:
1. Acknowledges the conversation context if relevant
2. Answers the current query clearly
3. Maintains continuity with previous discussions
4. Includes actionable recommendations

Response:"""
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def summarize_history(self, history):
        """Summarize conversation history"""
        summary_lines = []
        for msg in history:
            summary_lines.append(f"User: {msg['query']}")
            summary_lines.append(f"Advisor: {msg['response'][:200]}...")
        return "\n".join(summary_lines)
```

---

## 👤 User Profiler

```python
class UserProfiler:
    """Manage user profiles and preferences"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_profile(self, user_id):
        """Get user profile"""
        profile = self.db.query(
            "SELECT * FROM user_profiles WHERE user_id = %s", 
            (user_id,)
        )
        return profile
    
    def assess_risk_profile(self, user_id):
        """Conduct risk profile assessment"""
        questions = [
            {
                'question': 'How would you react to a 20% portfolio loss?',
                'options': [
                    {'text': 'Sell immediately', 'score': 1},
                    {'text': 'Hold and wait', 'score': 2},
                    {'text': 'Buy more at lower price', 'score': 3}
                ]
            },
            {
                'question': 'What is your investment time horizon?',
                'options': [
                    {'text': 'Less than 1 year', 'score': 1},
                    {'text': '1-5 years', 'score': 2},
                    {'text': 'More than 5 years', 'score': 3}
                ]
            },
            # ... more questions
        ]
        
        # Conduct questionnaire
        total_score = self.conduct_questionnaire(questions)
        
        # Determine profile
        if total_score <= 4:
            profile = 'conservative'
        elif total_score <= 7:
            profile = 'moderate'
        else:
            profile = 'aggressive'
        
        # Save to DB
        self.update_profile(user_id, {'risk_profile': profile})
        
        return profile
    
    def update_preferences(self, user_id, preferences):
        """Update user preferences"""
        self.db.execute(
            "UPDATE user_profiles SET preferences = %s WHERE user_id = %s",
            (json.dumps(preferences), user_id)
        )
    
    def track_interaction(self, user_id, query, response):
        """Log user interactions"""
        self.db.execute(
            """INSERT INTO interactions (user_id, query, response, timestamp) 
               VALUES (%s, %s, %s, %s)""",
            (user_id, query, response, datetime.now())
        )
```

---

## 🔌 REST API

### **FastAPI Backend**

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="LLM Financial Advisor API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize advisor
advisor = FinancialAdvisor()

# Request/Response models
class QueryRequest(BaseModel):
    user_id: str
    query: str
    include_explanations: bool = True

class AdviceResponse(BaseModel):
    user_id: str
    query: str
    recommendation: str
    confidence: float
    reasoning: list
    risks: list
    action_items: list
    detailed_analysis: dict
    sources: list
    timestamp: datetime

# Endpoints
@app.post("/api/advisor/advise", response_model=AdviceResponse)
async def get_advice(request: QueryRequest):
    """Get investment advice"""
    try:
        result = advisor.advise(request.query, request.user_id)
        return AdviceResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/advisor/history/{user_id}")
async def get_history(user_id: str, limit: int = 10):
    """Get user's conversation history"""
    history = advisor.conversation_manager.get_context(user_id, limit)
    return {"user_id": user_id, "history": history}

@app.post("/api/advisor/clear/{user_id}")
async def clear_history(user_id: str):
    """Clear user's conversation history"""
    advisor.conversation_manager.clear_history(user_id)
    return {"message": "History cleared", "user_id": user_id}

@app.post("/api/advisor/risk-assessment/{user_id}")
async def conduct_risk_assessment(user_id: str):
    """Conduct risk profile assessment"""
    profile = advisor.user_profiler.assess_risk_profile(user_id)
    return {"user_id": user_id, "risk_profile": profile}

@app.get("/api/advisor/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}
```

---

## 💬 Example Workflows

### **Workflow 1: Single Stock Query**

```python
# User: "Should I buy GP stock?"

result = advisor.advise(
    user_query="Should I buy GP stock?",
    user_id="user_123"
)

# Output structure:
{
    "recommendation": "BUY",
    "confidence": 78,
    "reasoning": [
        "Strong predicted return of +8.4% over next 5 days",
        "Positive news sentiment (compound: 0.65)",
        "Recent earnings beat expectations by 12%",
        "RSI indicates oversold conditions",
        "Matches your moderate risk profile"
    ],
    "risks": [
        "Market volatility currently 28%",
        "Sector rotation risk",
        "Quarterly results pending"
    ],
    "action_items": [
        "Allocate 30-40% of intended investment",
        "Set stop-loss at -8%",
        "Monitor quarterly results (Aug 15)",
        "Review position in 2 weeks"
    ],
    "time_horizon": "3-6 months",
    "detailed_analysis": {
        "prediction": {...},
        "sentiment": {...},
        "risk_metrics": {...}
    }
}
```

### **Workflow 2: Portfolio Recommendation**

```python
# User: "Build me a portfolio with 200,000 BDT for moderate risk"

result = advisor.advise(
    user_query="Build me a portfolio with 200,000 BDT for moderate risk",
    user_id="user_123"
)

# Process:
# 1. Orchestrator identifies: portfolio optimization
# 2. Agents activated: prediction, risk, portfolio
# 3. Portfolio agent optimizes allocation
# 4. Advisor synthesizes recommendation
```

### **Workflow 3: Follow-up Question**

```python
# Turn 1:
result1 = advisor.advise("Should I buy GP?", user_id)

# Turn 2 (context-aware):
result2 = advisor.advise(
    "What about the risks you mentioned?",  # Understands context
    user_id
)
```

---

## 🧠 Advanced Prompt Engineering

### **Chain-of-Thought Reasoning**

```python
class ChainOfThoughtAdvisor:
    """Advisor using chain-of-thought prompting"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_advice_with_reasoning(self, analysis):
        """Generate advice with explicit reasoning steps"""
        prompt = f"""Analyze this investment opportunity step-by-step.

Data:
{json.dumps(analysis, indent=2)}

Think through this carefully:

Step 1: Analyze the prediction
- What does the forecast suggest?
- How reliable is this prediction?
- What's the risk level?

Step 2: Consider market sentiment
- What is news sentiment telling us?
- Are there any major events?
- How does sentiment align with prediction?

Step 3: Evaluate company fundamentals
- What do the financials show?
- Is the company healthy?
- Any red flags?

Step 4: Assess user fit
- Does this match the user's risk profile?
- Can they handle the volatility?
- What are their goals?

Step 5: Make recommendation
- Buy, Sell, or Hold?
- What confidence level?
- What position size?
- What time horizon?

Provide your step-by-step analysis and final recommendation.
"""
        
        response = self.llm.invoke(prompt)
        return self.parse_chain_of_thought(response.content)
    
    def parse_chain_of_thought(self, response):
        """Parse chain-of-thought response"""
        steps = {}
        current_step = None
        
        for line in response.split('\n'):
            if line.startswith('Step'):
                current_step = line
                steps[current_step] = []
            elif current_step:
                steps[current_step].append(line)
        
        return steps
```

### **Few-Shot Learning**

```python
FEW_SHOT_EXAMPLES = [
    {
        'query': 'Should I buy GP?',
        'analysis': {
            'prediction': '+8.4% return',
            'sentiment': 'Positive (0.65)',
            'risk': 'Moderate'
        },
        'response': """Based on the analysis, I recommend BUYING GP with 78% confidence.

Reasoning:
1. Strong predicted return of 8.4% aligns with positive sentiment
2. Recent earnings beat supports bullish outlook
3. RSI suggests oversold conditions (entry point)

Risks: Market volatility, pending earnings

Action: Allocate 30-40%, stop-loss at -8%"""
    },
    # More examples...
]
```

---

## 🛡️ Safety & Ethics

```python
class SafetyChecker:
    """Ensure advice meets ethical standards"""
    
    def __init__(self):
        self.disclaimers = [
            "This is not financial advice. Always consult with a qualified advisor.",
            "Past performance does not guarantee future results.",
            "Investments carry risk of loss."
        ]
    
    def add_disclaimers(self, response):
        """Add required disclaimers"""
        disclaimer_text = "\n\n⚠️ DISCLAIMER:\n" + "\n".join(
            f"- {d}" for d in self.disclaimers
        )
        return response + disclaimer_text
    
    def check_for_misleading_claims(self, response):
        """Check for overpromising language"""
        red_flags = [
            'guaranteed', 'risk-free', 'sure thing', 
            'can\'t lose', '100% certain'
        ]
        
        for flag in red_flags:
            if flag.lower() in response.lower():
                return False, f"Contains misleading claim: '{flag}'"
        
        return True, "OK"
    
    def validate_confidence_score(self, confidence):
        """Ensure confidence is realistic"""
        if confidence > 95:
            return "Very high confidence - please verify the basis"
        elif confidence < 30:
            return "Low confidence - consider broader analysis"
        return "Appropriate confidence level"
```

---

## 📊 Advisor Performance Monitoring

```python
class AdvisorMonitor:
    """Monitor advisor performance and quality"""
    
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'avg_response_time': 0,
            'user_satisfaction': [],
            'recommendation_accuracy': []
        }
    
    def log_query(self, query, response, response_time, user_feedback=None):
        """Log advisor query and feedback"""
        self.metrics['total_queries'] += 1
        
        # Update average response time
        n = self.metrics['total_queries']
        self.metrics['avg_response_time'] = (
            (self.metrics['avg_response_time'] * (n-1) + response_time) / n
        )
        
        if user_feedback:
            self.metrics['user_satisfaction'].append(user_feedback)
    
    def track_recommendation_accuracy(self, recommendation, actual_outcome):
        """Track if recommendations were correct"""
        # Compare recommendation with actual stock movement
        self.metrics['recommendation_accuracy'].append({
            'recommendation': recommendation,
            'outcome': actual_outcome,
            'date': datetime.now()
        })
    
    def generate_performance_report(self):
        """Generate performance report"""
        report = {
            'total_queries': self.metrics['total_queries'],
            'avg_response_time_sec': self.metrics['avg_response_time'],
            'satisfaction_rate': np.mean(self.metrics['user_satisfaction']) 
                                if self.metrics['user_satisfaction'] else 0,
            'recommendation_accuracy': self.calculate_accuracy()
        }
        return report
    
    def calculate_accuracy(self):
        """Calculate recommendation accuracy"""
        if not self.metrics['recommendation_accuracy']:
            return 0
        
        correct = sum(
            1 for r in self.metrics['recommendation_accuracy']
            if r['recommendation'] == r['outcome']
        )
        return correct / len(self.metrics['recommendation_accuracy'])
```

---

## 📂 Project Structure

```
advisor/
├── core/
│   ├── financial_advisor.py
│   ├── conversation_manager.py
│   └── context_responder.py
├── user/
│   ├── user_profiler.py
│   ├── risk_assessment.py
│   └── preferences.py
├── api/
│   ├── main.py
│   ├── routes.py
│   └── middleware.py
├── prompts/
│   ├── advisor_prompts.py
│   ├── chain_of_thought.py
│   └── few_shot_examples.py
├── safety/
│   ├── disclaimers.py
│   ├── ethics_checker.py
│   └── validator.py
├── monitoring/
│   ├── performance_tracker.py
│   └── quality_metrics.py
└── tests/
    ├── test_advisor.py
    ├── test_conversation.py
    └── test_safety.py
```

---

## ✅ Success Criteria

- [ ] All components integrated successfully
- [ ] Conversational interface working
- [ ] Multi-turn dialogue tested
- [ ] REST API operational
- [ ] Response time < 5 seconds
- [ ] Context maintained across turns
- [ ] Safety disclaimers present
- [ ] User profiling functional
- [ ] Performance monitoring active
- [ ] Documentation complete

---

## 🛠️ Tools & Libraries

- **FastAPI**: REST API framework
- **OpenAI API**: LLM backbone
- **LangChain**: LLM orchestration
- **Pydantic**: Data validation
- **Redis**: Conversation caching
- **PostgreSQL**: User data storage

---

## 💡 Best Practices

1. **Validate inputs** before processing
2. **Handle errors gracefully** with fallback responses
3. **Add safety disclaimers** to all advice
4. **Log everything** for debugging
5. **Monitor performance** continuously
6. **Version control** prompts and configurations
7. **A/B test** prompt variations
8. **Rate limit** API calls

---

## 🎯 Example End-to-End Interaction

```python
# Initialize
advisor = FinancialAdvisor()
user_id = "user_123"

# Conversation flow
queries = [
    "Hello",
    "Can you tell me about GP stock?",
    "Should I invest in it?",
    "What are the risks?",
    "What's a good entry point?",
    "Build me a portfolio with 100,000 BDT"
]

for query in queries:
    print(f"\n👤 User: {query}")
    result = advisor.advise(query, user_id)
    print(f"🤖 Advisor: {result['recommendation']}")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Key Points: {result['reasoning'][:3]}")
```

---

**Next Phase**: Phase 13 — Dashboard

**Last Updated**: 2026-08-13
