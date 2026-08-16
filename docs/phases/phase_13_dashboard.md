# PHASE 13 — Dashboard

**Duration**: 2 Weeks  
**Started**: Week 25  
**Status**: 📝 Pending  
**Goal**: Build user-facing web application

---

## 🎯 Objectives

1. Build responsive Next.js application
2. Create intuitive UI/UX
3. Implement real-time data visualization
4. Build interactive chatbot interface
5. Deploy production-ready application

---

## 🛠️ Technology Stack

```
Frontend:
├── Next.js 14 (React framework)
├── TypeScript (type safety)
├── Tailwind CSS (styling)
├── Recharts (data visualization)
├── Zustand (state management)
└── Axios (API calls)

Backend:
├── FastAPI (Python)
├── PostgreSQL (database)
├── Redis (caching)
└── WebSocket (real-time)

Deployment:
├── Docker (containerization)
├── Vercel (frontend hosting)
└── AWS/GCP (backend hosting)
```

---

## 📁 Project Structure

```
frontend/
├── public/
├── src/
│   ├── pages/
│   │   ├── _app.tsx
│   │   ├── index.tsx           # Dashboard
│   │   ├── stocks/
│   │   │   └── [code].tsx      # Stock analysis page
│   │   ├── portfolio.tsx       # Portfolio management
│   │   ├── advisor.tsx         # Chatbot
│   │   ├── reports.tsx         # Research reports
│   │   └── api/                # API routes
│   ├── components/
│   │   ├── Layout/
│   │   ├── Charts/
│   │   ├── Cards/
│   │   ├── Forms/
│   │   └── Chatbot/
│   ├── hooks/
│   ├── services/
│   ├── store/
│   ├── types/
│   └── utils/
├── styles/
├── package.json
└── next.config.js
```

---

## 🏠 Page 1: Dashboard

```typescript
// pages/index.tsx
import { useEffect, useState } from 'react';
import { Line, Bar, Pie } from 'recharts';
import MarketOverview from '@/components/Dashboard/MarketOverview';
import StockCard from '@/components/Cards/StockCard';
import NewsFeed from '@/components/Dashboard/NewsFeed';
import { stockService } from '@/services/stockService';

export default function Dashboard() {
  const [marketData, setMarketData] = useState(null);
  const [topStocks, setTopStocks] = useState([]);
  const [news, setNews] = useState([]);
  
  useEffect(() => {
    loadDashboardData();
  }, []);
  
  const loadDashboardData = async () => {
    const [market, stocks, newsData] = await Promise.all([
      stockService.getMarketIndices(),
      stockService.getTopStocks(),
      stockService.getLatestNews()
    ]);
    setMarketData(market);
    setTopStocks(stocks);
    setNews(newsData);
  };
  
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">DSE Market Dashboard</h1>
      
      {/* Market Overview */}
      <MarketOverview data={marketData} />
      
      {/* Top Stocks Grid */}
      <section className="mt-12">
        <h2 className="text-2xl font-semibold mb-4">Top Performing Stocks</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {topStocks.map(stock => (
            <StockCard key={stock.code} stock={stock} />
          ))}
        </div>
      </section>
      
      {/* News Feed */}
      <NewsFeed articles={news} />
    </div>
  );
}
```

### **Components**

```typescript
// components/Dashboard/MarketOverview.tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

export default function MarketOverview({ data }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <IndexCard 
        name="DSEX" 
        value={data?.dsex} 
        change={data?.dsex_change} 
      />
      <IndexCard 
        name="DS30" 
        value={data?.ds30} 
        change={data?.ds30_change} 
      />
      <IndexCard 
        name="DSES" 
        value={data?.dses} 
        change={data?.dses_change} 
      />
      
      <div className="col-span-3 bg-white rounded-lg shadow p-6">
        <h3 className="text-xl font-semibold mb-4">DSEX Trend (30 Days)</h3>
        <LineChart width={1000} height={300} data={data?.historical}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke="#8884d8" />
        </LineChart>
      </div>
    </div>
  );
}

// components/Cards/StockCard.tsx
import Link from 'next/link';

export default function StockCard({ stock }) {
  const isPositive = stock.change >= 0;
  
  return (
    <Link href={`/stocks/${stock.code}`}>
      <div className="bg-white rounded-lg shadow p-4 hover:shadow-lg transition-shadow cursor-pointer">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-bold">{stock.code}</h3>
          <span className={`text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {isPositive ? '▲' : '▼'} {Math.abs(stock.change).toFixed(2)}%
          </span>
        </div>
        <p className="text-sm text-gray-600 mt-1">{stock.name}</p>
        <p className="text-2xl font-semibold mt-3">৳{stock.price.toFixed(2)}</p>
        <p className="text-xs text-gray-500 mt-1">
          Vol: {(stock.volume / 1000).toFixed(0)}K
        </p>
      </div>
    </Link>
  );
}
```

---

## 📊 Page 2: Stock Analysis

```typescript
// pages/stocks/[code].tsx
import { useRouter } from 'next/router';
import { useState, useEffect } from 'react';
import { Line, Bar, ComposedChart } from 'recharts';
import StockChart from '@/components/Charts/StockChart';
import PredictionPanel from '@/components/Stocks/PredictionPanel';
import SentimentPanel from '@/components/Stocks/SentimentPanel';
import FundamentalsPanel from '@/components/Stocks/FundamentalsPanel';
import TechnicalIndicators from '@/components/Stocks/TechnicalIndicators';

export default function StockAnalysis() {
  const router = useRouter();
  const { code } = router.query;
  
  const [stockData, setStockData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [sentiment, setSentiment] = useState(null);
  const [fundamentals, setFundamentals] = useState(null);
  const [timeframe, setTimeframe] = useState('1Y');
  
  useEffect(() => {
    if (code) {
      loadStockData(code as string);
    }
  }, [code, timeframe]);
  
  const loadStockData = async (stockCode: string) => {
    const [data, pred, sent, fund] = await Promise.all([
      stockService.getHistorical(stockCode, timeframe),
      stockService.getPredictions(stockCode),
      stockService.getSentiment(stockCode),
      stockService.getFundamentals(stockCode)
    ]);
    setStockData(data);
    setPredictions(pred);
    setSentiment(sent);
    setFundamentals(fund);
  };
  
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-4xl font-bold">{code}</h1>
          <p className="text-gray-600">{stockData?.name}</p>
        </div>
        <div className="text-right">
          <p className="text-3xl font-bold">৳{stockData?.current_price}</p>
          <p className={`text-lg ${stockData?.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {stockData?.change >= 0 ? '+' : ''}{stockData?.change}%
          </p>
        </div>
      </div>
      
      {/* Timeframe selector */}
      <div className="flex gap-2 mb-6">
        {['1M', '3M', '6M', '1Y', '5Y', 'ALL'].map(tf => (
          <button
            key={tf}
            onClick={() => setTimeframe(tf)}
            className={`px-4 py-2 rounded ${
              timeframe === tf 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            {tf}
          </button>
        ))}
      </div>
      
      {/* Price Chart with Predictions */}
      <StockChart data={stockData} predictions={predictions} />
      
      {/* Analysis Panels Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
        <PredictionPanel predictions={predictions} />
        <SentimentPanel sentiment={sentiment} />
        <FundamentalsPanel fundamentals={fundamentals} />
        <TechnicalIndicators data={stockData} />
      </div>
      
      {/* Ask Advisor */}
      <div className="mt-8">
        <button 
          onClick={() => router.push(`/advisor?stock=${code}`)}
          className="w-full bg-blue-600 text-white py-4 rounded-lg hover:bg-blue-700"
        >
          💬 Ask Advisor about {code}
        </button>
      </div>
    </div>
  );
}
```

### **Stock Chart Component**

```typescript
// components/Charts/StockChart.tsx
import { ComposedChart, Line, Area, Bar, XAxis, YAxis, 
         CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function StockChart({ data, predictions }) {
  // Merge historical and predicted data
  const chartData = [
    ...data?.historical?.map(d => ({ ...d, type: 'historical' })),
    ...predictions?.forecast?.map(d => ({ ...d, type: 'predicted' }))
  ];
  
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          
          {/* Historical price */}
          <Area 
            yAxisId="left"
            type="monotone" 
            dataKey="close" 
            fill="#8884d8" 
            fillOpacity={0.3}
            stroke="#8884d8"
            name="Historical Price"
          />
          
          {/* Predicted price */}
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="predicted_close" 
            stroke="#82ca9d" 
            strokeDasharray="5 5"
            strokeWidth={2}
            name="Predicted Price"
          />
          
          {/* Volume */}
          <Bar 
            yAxisId="right"
            dataKey="volume" 
            fill="#ffc658"
            fillOpacity={0.3}
            name="Volume"
          />
          
          {/* Confidence interval */}
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="upper_bound" 
            stroke="#ddd" 
            strokeDasharray="3 3"
            dot={false}
            name="Upper Bound"
          />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="lower_bound" 
            stroke="#ddd" 
            strokeDasharray="3 3"
            dot={false}
            name="Lower Bound"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
```

---

## 💬 Page 3: AI Advisor Chatbot

```typescript
// pages/advisor.tsx
import { useState, useRef, useEffect } from 'react';
import ChatMessage from '@/components/Chatbot/ChatMessage';
import ChatInput from '@/components/Chatbot/ChatInput';
import SuggestedQueries from '@/components/Chatbot/SuggestedQueries';
import { advisorService } from '@/services/advisorService';

interface Message {
  role: 'user' | 'advisor';
  content: string;
  timestamp: Date;
  data?: any;
}

export default function AdvisorPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'advisor',
      content: 'Hello! I\'m your AI financial advisor for the Bangladesh stock market. How can I help you today?',
      timestamp: new Date()
    }
  ]);
  
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [userId] = useState(`user_${Date.now()}`);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      const response = await advisorService.getAdvice(userId, input);
      
      const advisorMessage: Message = {
        role: 'advisor',
        content: response.recommendation,
        timestamp: new Date(),
        data: response
      };
      
      setMessages(prev => [...prev, advisorMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'advisor',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  const handleSuggestion = (suggestion: string) => {
    setInput(suggestion);
  };
  
  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="bg-white rounded-lg shadow-lg h-[700px] flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-t-lg">
          <h2 className="text-2xl font-bold">AI Financial Advisor</h2>
          <p className="text-sm opacity-90">Powered by LLMs and Multi-Agent System</p>
        </div>
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, idx) => (
            <ChatMessage key={idx} message={msg} />
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-200 rounded-lg p-3">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        {/* Suggested queries (only show initially) */}
        {messages.length === 1 && (
          <SuggestedQueries onSelect={handleSuggestion} />
        )}
        
        {/* Input */}
        <ChatInput 
          value={input}
          onChange={setInput}
          onSend={sendMessage}
          disabled={isLoading}
        />
      </div>
    </div>
  );
}
```

### **Chat Message Component**

```typescript
// components/Chatbot/ChatMessage.tsx
import { useState } from 'react';

export default function ChatMessage({ message }) {
  const [showDetails, setShowDetails] = useState(false);
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[70%] rounded-lg p-4 ${
        isUser 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-100 text-gray-900'
      }`}>
        <p className="whitespace-pre-wrap">{message.content}</p>
        
        {/* Show detailed analysis for advisor messages */}
        {!isUser && message.data && (
          <div className="mt-3 pt-3 border-t border-gray-300">
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-sm text-blue-600 hover:underline"
            >
              {showDetails ? '▼ Hide' : '▶ Show'} Details
            </button>
            
            {showDetails && (
              <div className="mt-3 space-y-2 text-sm">
                {message.data.confidence && (
                  <div>
                    <strong>Confidence:</strong> {message.data.confidence}%
                  </div>
                )}
                
                {message.data.reasoning && (
                  <div>
                    <strong>Reasoning:</strong>
                    <ul className="list-disc list-inside ml-2 mt-1">
                      {message.data.reasoning.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {message.data.risks && (
                  <div>
                    <strong>Risks:</strong>
                    <ul className="list-disc list-inside ml-2 mt-1">
                      {message.data.risks.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {message.data.action_items && (
                  <div>
                    <strong>Action Items:</strong>
                    <ul className="list-disc list-inside ml-2 mt-1">
                      {message.data.action_items.map((a, i) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        
        <p className="text-xs opacity-70 mt-2">
          {new Date(message.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
}
```

---

## 💼 Page 4: Portfolio Management

```typescript
// pages/portfolio.tsx
import { useState, useEffect } from 'react';
import { Pie, Bar } from 'recharts';
import PortfolioSummary from '@/components/Portfolio/PortfolioSummary';
import HoldingsTable from '@/components/Portfolio/HoldingsTable';
import AllocationChart from '@/components/Portfolio/AllocationChart';
import PerformanceChart from '@/components/Portfolio/PerformanceChart';
import RebalancingSuggestions from '@/components/Portfolio/RebalancingSuggestions';

export default function PortfolioPage() {
  const [portfolio, setPortfolio] = useState(null);
  const [performance, setPerformance] = useState(null);
  
  useEffect(() => {
    loadPortfolio();
  }, []);
  
  const loadPortfolio = async () => {
    const data = await portfolioService.getPortfolio();
    setPortfolio(data);
    setPerformance(data.performance);
  };
  
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">My Portfolio</h1>
      
      {/* Summary Cards */}
      <PortfolioSummary portfolio={portfolio} />
      
      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
        <AllocationChart allocation={portfolio?.allocation} />
        <PerformanceChart performance={performance} />
      </div>
      
      {/* Holdings */}
      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-4">Holdings</h2>
        <HoldingsTable holdings={portfolio?.holdings} />
      </div>
      
      {/* Rebalancing Suggestions */}
      <RebalancingSuggestions suggestions={portfolio?.rebalancing} />
      
      {/* Optimize Button */}
      <button className="mt-8 w-full bg-blue-600 text-white py-4 rounded-lg hover:bg-blue-700">
        Optimize Portfolio
      </button>
    </div>
  );
}
```

---

## 🎨 Styling (Tailwind)

```typescript
// tailwind.config.js
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx}',
    './src/components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        success: '#10b981',
        danger: '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
```

### **Global Styles**

```css
/* styles/globals.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer components {
  .btn-primary {
    @apply bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors;
  }
  
  .card {
    @apply bg-white rounded-lg shadow p-6;
  }
  
  .badge-green {
    @apply bg-green-100 text-green-800 px-2 py-1 rounded text-xs;
  }
  
  .badge-red {
    @apply bg-red-100 text-red-800 px-2 py-1 rounded text-xs;
  }
}
```

---

## 🔌 API Services

```typescript
// services/api.ts
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);
```

```typescript
// services/stockService.ts
import { apiClient } from './api';

export const stockService = {
  async getMarketIndices() {
    const response = await apiClient.get('/api/market/indices');
    return response.data;
  },
  
  async getHistorical(code: string, timeframe: string) {
    const response = await apiClient.get(
      `/api/stocks/${code}/historical?timeframe=${timeframe}`
    );
    return response.data;
  },
  
  async getPredictions(code: string) {
    const response = await apiClient.get(`/api/stocks/${code}/predictions`);
    return response.data;
  },
  
  async getSentiment(code: string) {
    const response = await apiClient.get(`/api/stocks/${code}/sentiment`);
    return response.data;
  },
  
  async getFundamentals(code: string) {
    const response = await apiClient.get(`/api/stocks/${code}/fundamentals`);
    return response.data;
  },
  
  async getTopStocks(limit: number = 20) {
    const response = await apiClient.get(`/api/stocks/top?limit=${limit}`);
    return response.data;
  },
  
  async getLatestNews(limit: number = 10) {
    const response = await apiClient.get(`/api/news/latest?limit=${limit}`);
    return response.data;
  },
};
```

```typescript
// services/advisorService.ts
import { apiClient } from './api';

export const advisorService = {
  async getAdvice(userId: string, query: string) {
    const response = await apiClient.post('/api/advisor/advise', {
      user_id: userId,
      query: query,
    });
    return response.data;
  },
  
  async getHistory(userId: string) {
    const response = await apiClient.get(`/api/advisor/history/${userId}`);
    return response.data;
  },
  
  async clearHistory(userId: string) {
    const response = await apiClient.post(`/api/advisor/clear/${userId}`);
    return response.data;
  },
};
```

---

## 📱 Responsive Design

```typescript
// hooks/useMediaQuery.ts
import { useState, useEffect } from 'react';

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);
  
  useEffect(() => {
    const media = window.matchMedia(query);
    setMatches(media.matches);
    
    const listener = (e: MediaQueryListEvent) => setMatches(e.matches);
    media.addEventListener('change', listener);
    
    return () => media.removeEventListener('change', listener);
  }, [query]);
  
  return matches;
}

// Usage:
// const isMobile = useMediaQuery('(max-width: 768px)');
```

---

## 🐳 Docker Deployment

```dockerfile
# Dockerfile.frontend
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV production
COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend
  
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/dse
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=dse
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
```

---

## ✅ Success Criteria

- [ ] All pages built and functional
- [ ] Responsive design working on mobile/tablet/desktop
- [ ] Charts rendering correctly
- [ ] Chatbot interface smooth
- [ ] API integration successful
- [ ] Loading states implemented
- [ ] Error handling in place
- [ ] Performance optimized (<3s load time)
- [ ] Docker deployment working
- [ ] SSL/HTTPS configured
- [ ] User authentication (optional)

---

## 🛠️ Tools & Libraries

- **Next.js 14**: React framework with SSR/SSG
- **TypeScript**: Type safety
- **Tailwind CSS**: Utility-first styling
- **Recharts**: Data visualization
- **Zustand**: State management
- **Axios**: HTTP client
- **React Query**: Data fetching/caching
- **NextAuth.js**: Authentication (optional)
- **Docker**: Containerization
- **Vercel**: Deployment platform

---

## 💡 Best Practices

1. **Code splitting** for faster load times
2. **Lazy loading** images and components
3. **Memoization** for expensive computations
4. **Error boundaries** for graceful failures
5. **SEO optimization** with Next.js metadata
6. **Accessibility** (ARIA labels, keyboard nav)
7. **Performance monitoring** (Lighthouse, Web Vitals)

---

**Next Phase**: Phase 14 — Research Paper

**Last Updated**: 2026-08-13
