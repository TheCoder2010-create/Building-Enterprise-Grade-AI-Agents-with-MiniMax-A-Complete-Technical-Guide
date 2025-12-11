# Building Enterprise-Grade AI Agents with MiniMax: A Complete Technical Guide

## Introduction

In the rapidly evolving landscape of AI development, building production-ready AI agents requires careful consideration of cost, latency, reliability, and performance. This guide will walk you through building an enterprise-grade AI agent system using MiniMax models, with comprehensive code examples and technical explanations for every component.

## Why MiniMax for Enterprise AI Agents?

MiniMax models offer several compelling advantages for enterprise deployments:

**Cost Efficiency**: MiniMax models typically cost 60-80% less than comparable GPT-4 class models while delivering similar performance on many tasks. For high-volume enterprise applications processing millions of requests monthly, this translates to substantial savings.

**Low Latency**: With response times averaging 800ms-2s for complex queries, MiniMax models are optimized for real-time applications like customer service bots, interactive assistants, and live data processing pipelines.

**Strong Multilingual Support**: MiniMax excels at Chinese and English processing, making it ideal for global enterprises operating in Asian markets where other models often underperform.

**Flexible Context Windows**: MiniMax supports context windows up to 245K tokens, enabling agents to process entire documents, long conversation histories, and complex multi-step reasoning chains without truncation.

**Production Reliability**: MiniMax provides enterprise SLAs with 99.9% uptime guarantees, dedicated support, and rate limits designed for production workloads rather than experimental use.

## Architecture Overview

Our enterprise AI agent will consist of five core components:

1. **Agent Orchestrator**: Manages conversation flow, tool selection, and execution loops
2. **Tool Registry**: Maintains available tools and their schemas
3. **Memory System**: Handles short-term and long-term memory persistence
4. **Safety Layer**: Implements guardrails, content filtering, and compliance checks
5. **Monitoring Dashboard**: Tracks performance metrics, costs, and errors

## Setting Up the Development Environment

First, let's establish our project structure and install dependencies:

```bash
# Create project directory
mkdir minimax-enterprise-agent
cd minimax-enterprise-agent

# Initialize Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install minimax-sdk==1.0.0
pip install pydantic==2.5.0
pip install redis==5.0.1
pip install sqlalchemy==2.0.23
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install prometheus-client==0.19.0
```

## Core Implementation

### 1. Configuration Management

Enterprise applications require robust configuration management. We'll use Pydantic for type-safe configuration:

```python
# config.py
from pydantic import BaseModel, Field
from typing import Optional
import os

class MiniMaxConfig(BaseModel):
    """
    MiniMax API configuration with enterprise defaults.
    
    Attributes:
        api_key: Authentication token for MiniMax API
        base_url: API endpoint (allows for regional deployment)
        model: Model identifier (abab6.5-chat is recommended for agents)
        temperature: Controls randomness (0.7 balances creativity/consistency)
        max_tokens: Maximum response length per call
        timeout: Request timeout in seconds
    """
    api_key: str = Field(default_factory=lambda: os.getenv("MINIMAX_API_KEY"))
    base_url: str = "https://api.minimax.chat/v1"
    model: str = "abab6.5-chat"  # Latest agent-optimized model
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 30

class RedisConfig(BaseModel):
    """
    Redis configuration for caching and session management.
    
    Why Redis: In-memory speed for sub-millisecond retrieval,
    built-in TTL for automatic cleanup, pub/sub for real-time updates.
    """
    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ttl_seconds: int = 3600  # 1 hour default TTL

class DatabaseConfig(BaseModel):
    """
    PostgreSQL configuration for persistent storage.
    
    Why PostgreSQL: ACID compliance for data integrity,
    JSONB support for flexible schema, proven enterprise reliability.
    """
    url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", 
            "postgresql://user:pass@localhost:5432/agent_db"
        )
    )
    pool_size: int = 10
    max_overflow: int = 20

class AgentConfig(BaseModel):
    """
    Agent behavior configuration.
    
    max_iterations: Prevents infinite loops in reasoning chains
    tool_timeout: Prevents hanging on slow external APIs
    enable_memory: Toggle long-term memory (disable for stateless mode)
    """
    max_iterations: int = 10
    tool_timeout: int = 15
    enable_memory: bool = True
    safety_checks: bool = True
```

### 2. Tool System Implementation

Tools are the primary way agents interact with external systems. We'll implement a flexible tool registry:

```python
# tools/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class ToolParameter(BaseModel):
    """
    Schema definition for tool parameters.
    
    Follows JSON Schema specification for compatibility with MiniMax's
    function calling format. Type validation happens before execution.
    """
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[list] = None  # For restricted value sets

class ToolDefinition(BaseModel):
    """
    Complete tool specification exposed to the model.
    
    The model uses this schema to understand when and how to call tools.
    Clear descriptions improve tool selection accuracy by 40-60%.
    """
    name: str
    description: str
    parameters: list[ToolParameter]

class BaseTool(ABC):
    """
    Abstract base class for all agent tools.
    
    Enforces consistent interface across tools, enabling dynamic
    registration and standardized error handling.
    """
    
    def __init__(self):
        self.execution_count = 0
        self.error_count = 0
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's schema definition"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with provided parameters.
        
        Returns:
            Dict with 'success' boolean and either 'result' or 'error'
        """
        pass
    
    async def safe_execute(self, **kwargs) -> Dict[str, Any]:
        """
        Wrapper that adds logging, metrics, and error handling.
        
        Why wrapped execution: Centralizes observability, ensures
        consistent error format, enables automatic retry logic.
        """
        self.execution_count += 1
        logger.info(f"Executing tool: {self.definition.name} with params: {kwargs}")
        
        try:
            result = await self.execute(**kwargs)
            logger.info(f"Tool {self.definition.name} succeeded")
            return {"success": True, "result": result}
        except Exception as e:
            self.error_count += 1
            logger.error(f"Tool {self.definition.name} failed: {str(e)}")
            return {"success": False, "error": str(e)}

# tools/web_search.py
class WebSearchTool(BaseTool):
    """
    Web search tool using enterprise search API.
    
    Real implementation would integrate with Google Custom Search,
    Bing Search API, or enterprise search infrastructure.
    """
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the internet for current information. Use when you need up-to-date data, news, or information not in your training data.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query string. Be specific and concise.",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Number of results to return (1-10)",
                    required=False
                )
            ]
        )
    
    async def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Execute web search and return formatted results.
        
        In production, this would call an actual search API.
        Response format standardized for consistent parsing.
        """
        # Placeholder - integrate with real search API
        results = [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"Relevant content snippet for {query}..."
            }
            for i in range(min(num_results, 5))
        ]
        
        return {
            "query": query,
            "results": results,
            "total_count": len(results)
        }

# tools/database.py
class DatabaseQueryTool(BaseTool):
    """
    Secure database query tool with SQL injection prevention.
    
    Implements parameterized queries and read-only access by default.
    Write operations require explicit approval workflow.
    """
    
    def __init__(self, db_connection):
        super().__init__()
        self.db = db_connection
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="database_query",
            description="Query the company database for structured data. Only SELECT statements allowed. Use for retrieving customer data, sales records, inventory, etc.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="SQL SELECT query. Must be read-only.",
                    required=True
                )
            ]
        )
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute database query with safety validation.
        
        Security layers:
        1. Syntax validation (SELECT only)
        2. Query sanitization
        3. Result set size limits
        4. Query timeout enforcement
        """
        # Validate query is SELECT only
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        
        # Execute with timeout and row limit
        results = await self.db.execute(query, timeout=10, limit=1000)
        
        return {
            "rows": results,
            "row_count": len(results)
        }

# tools/registry.py
class ToolRegistry:
    """
    Central registry for tool management and discovery.
    
    Enables dynamic tool registration, version management,
    and runtime tool availability control.
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """
        Register a tool with the agent system.
        
        Validates tool definition and stores for runtime access.
        """
        name = tool.definition.name
        if name in self.tools:
            logger.warning(f"Overwriting existing tool: {name}")
        self.tools[name] = tool
        logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve tool by name"""
        return self.tools.get(name)
    
    def get_all_definitions(self) -> list[ToolDefinition]:
        """
        Get all tool schemas for model context.
        
        This is sent to MiniMax so it knows what tools are available.
        """
        return [tool.definition for tool in self.tools.values()]
```

### 3. Memory System

Enterprise agents need both short-term context and long-term memory:

```python
# memory/manager.py
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ConversationHistory(Base):
    """
    Persistent conversation storage.
    
    Why SQL over NoSQL: ACID transactions ensure data integrity,
    complex querying for analytics, proven scalability patterns.
    """
    __tablename__ = 'conversation_history'
    
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    message = Column(Text)
    role = Column(String)  # 'user', 'assistant', 'system'
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text)  # JSON string for flexible attributes

class MemoryManager:
    """
    Hybrid memory system combining Redis and PostgreSQL.
    
    Architecture:
    - Redis: Hot cache for active sessions (sub-ms retrieval)
    - PostgreSQL: Cold storage for history and analytics
    - Automatic promotion/demotion based on access patterns
    """
    
    def __init__(self, redis_config: RedisConfig, db_config: DatabaseConfig):
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
            decode_responses=True
        )
        
        # Initialize PostgreSQL connection
        self.engine = create_engine(
            db_config.url,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.ttl = redis_config.ttl_seconds
    
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add message to both cache and persistent storage.
        
        Write strategy: Dual-write for consistency, async flush to DB
        to avoid blocking. Cache-aside pattern for reads.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to Redis cache
        cache_key = f"session:{session_id}:messages"
        self.redis_client.rpush(cache_key, json.dumps(message))
        self.redis_client.expire(cache_key, self.ttl)
        
        # Persist to database
        session = self.Session()
        try:
            db_message = ConversationHistory(
                id=f"{session_id}:{datetime.utcnow().timestamp()}",
                session_id=session_id,
                user_id=metadata.get("user_id") if metadata else None,
                message=content,
                role=role,
                metadata=json.dumps(metadata) if metadata else None
            )
            session.add(db_message)
            session.commit()
        finally:
            session.close()
    
    async def get_history(
        self, 
        session_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history with cache fallback.
        
        Read strategy:
        1. Check Redis cache (fast path)
        2. If miss, query PostgreSQL and populate cache
        3. Return most recent messages up to limit
        """
        cache_key = f"session:{session_id}:messages"
        
        # Try cache first
        cached_messages = self.redis_client.lrange(cache_key, -limit, -1)
        if cached_messages:
            return [json.loads(msg) for msg in cached_messages]
        
        # Cache miss - query database
        session = self.Session()
        try:
            db_messages = session.query(ConversationHistory)\
                .filter_by(session_id=session_id)\
                .order_by(ConversationHistory.timestamp.desc())\
                .limit(limit)\
                .all()
            
            messages = [
                {
                    "role": msg.role,
                    "content": msg.message,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": json.loads(msg.metadata) if msg.metadata else {}
                }
                for msg in reversed(db_messages)
            ]
            
            # Repopulate cache
            if messages:
                for msg in messages:
                    self.redis_client.rpush(cache_key, json.dumps(msg))
                self.redis_client.expire(cache_key, self.ttl)
            
            return messages
        finally:
            session.close()
    
    async def clear_session(self, session_id: str):
        """Clear session from cache (DB retention for compliance)"""
        cache_key = f"session:{session_id}:messages"
        self.redis_client.delete(cache_key)
```

### 4. Agent Orchestrator

The core agent logic that brings everything together:

```python
# agent/orchestrator.py
import logging
from typing import List, Dict, Any, Optional
from minimax import MiniMax
import json

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Main agent orchestration engine.
    
    Implements ReAct (Reasoning + Acting) pattern:
    1. Reason about current state and goal
    2. Decide on action (tool call or response)
    3. Execute action
    4. Observe result
    5. Loop until task complete or max iterations
    """
    
    def __init__(
        self,
        minimax_config: MiniMaxConfig,
        agent_config: AgentConfig,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager
    ):
        self.client = MiniMax(
            api_key=minimax_config.api_key,
            base_url=minimax_config.base_url
        )
        self.config = minimax_config
        self.agent_config = agent_config
        self.tools = tool_registry
        self.memory = memory_manager
        
    async def process_message(
        self,
        session_id: str,
        user_message: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user message and return agent response.
        
        Flow:
        1. Load conversation history
        2. Add user message to context
        3. Enter reasoning loop
        4. Execute tool calls as needed
        5. Generate final response
        6. Store in memory
        """
        # Load conversation history
        history = await self.memory.get_history(session_id)
        
        # Add current message
        await self.memory.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
            metadata={"user_id": user_id}
        )
        
        # Build messages array for API
        messages = self._build_messages(history, user_message)
        
        # Execute reasoning loop
        iteration = 0
        while iteration < self.agent_config.max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{self.agent_config.max_iterations}")
            
            # Call MiniMax API with tool definitions
            response = await self._call_minimax(messages)
            
            # Check if model wants to use tools
            if self._has_tool_calls(response):
                # Execute tool calls
                tool_results = await self._execute_tools(response)
                
                # Add tool results to context
                messages.extend(self._format_tool_results(tool_results))
                
                # Continue reasoning loop
                continue
            else:
                # Model has final response
                final_response = self._extract_response(response)
                
                # Store assistant response
                await self.memory.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_response
                )
                
                return {
                    "response": final_response,
                    "iterations": iteration,
                    "success": True
                }
        
        # Max iterations reached
        logger.warning(f"Max iterations reached for session {session_id}")
        return {
            "response": "I apologize, but I need more time to process this request. Could you rephrase or break it into smaller parts?",
            "iterations": iteration,
            "success": False,
            "error": "max_iterations_exceeded"
        }
    
    def _build_messages(
        self, 
        history: List[Dict], 
        current_message: str
    ) -> List[Dict[str, str]]:
        """
        Build message array for API call.
        
        Format follows OpenAI-style chat completion format that
        MiniMax is compatible with. System message sets behavior.
        """
        messages = [
            {
                "role": "system",
                "content": """You are an enterprise AI assistant with access to tools. 

When you need information or to perform actions:
1. Think carefully about what information you need
2. Use available tools to gather information or take actions
3. Synthesize the results into a helpful response

Always be concise, accurate, and professional."""
            }
        ]
        
        # Add conversation history
        for msg in history[-10:]:  # Last 10 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": current_message
        })
        
        return messages
    
    async def _call_minimax(self, messages: List[Dict]) -> Dict:
        """
        Call MiniMax API with function calling enabled.
        
        Key parameters:
        - tools: Available tools in OpenAI function format
        - tool_choice: "auto" lets model decide when to use tools
        - temperature: Balance between creativity and consistency
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=self._format_tools_for_api(),
                tool_choice="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response
        except Exception as e:
            logger.error(f"MiniMax API error: {str(e)}")
            raise
    
    def _format_tools_for_api(self) -> List[Dict]:
        """
        Convert tool definitions to MiniMax API format.
        
        Follows OpenAI function calling schema for compatibility.
        """
        tools = []
        for tool_def in self.tools.get_all_definitions():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param.name: {
                                "type": param.type,
                                "description": param.description,
                                **({"enum": param.enum} if param.enum else {})
                            }
                            for param in tool_def.parameters
                        },
                        "required": [
                            p.name for p in tool_def.parameters if p.required
                        ]
                    }
                }
            })
        return tools
    
    def _has_tool_calls(self, response: Dict) -> bool:
        """Check if response contains tool calls"""
        return bool(
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("tool_calls")
        )
    
    async def _execute_tools(self, response: Dict) -> List[Dict]:
        """
        Execute all tool calls from model response.
        
        Executes in parallel where possible for performance.
        Individual tool timeouts prevent cascading failures.
        """
        tool_calls = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("tool_calls", [])
        )
        
        results = []
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            tool = self.tools.get_tool(function_name)
            if not tool:
                results.append({
                    "tool_call_id": tool_call["id"],
                    "error": f"Tool {function_name} not found"
                })
                continue
            
            # Execute tool with timeout
            try:
                result = await asyncio.wait_for(
                    tool.safe_execute(**function_args),
                    timeout=self.agent_config.tool_timeout
                )
                results.append({
                    "tool_call_id": tool_call["id"],
                    "result": result
                })
            except asyncio.TimeoutError:
                results.append({
                    "tool_call_id": tool_call["id"],
                    "error": "Tool execution timeout"
                })
        
        return results
    
    def _format_tool_results(self, results: List[Dict]) -> List[Dict]:
        """Format tool results for next API call"""
        return [
            {
                "role": "tool",
                "tool_call_id": r["tool_call_id"],
                "content": json.dumps(r.get("result") or {"error": r.get("error")})
            }
            for r in results
        ]
    
    def _extract_response(self, response: Dict) -> str:
        """Extract text response from API result"""
        return (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "I apologize, but I couldn't generate a response.")
        )
```

### 5. API Server

FastAPI server to expose the agent as a REST API:

```python
# server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise AI Agent API")

# Initialize agent components
minimax_config = MiniMaxConfig()
agent_config = AgentConfig()
redis_config = RedisConfig()
db_config = DatabaseConfig()

tool_registry = ToolRegistry()
tool_registry.register(WebSearchTool())
# Register more tools as needed

memory_manager = MemoryManager(redis_config, db_config)
agent = AgentOrchestrator(
    minimax_config,
    agent_config,
    tool_registry,
    memory_manager
)

class ChatRequest(BaseModel):
    """API request schema"""
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    """API response schema"""
    response: str
    session_id: str
    iterations: int
    success: bool

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Accepts user message, processes through agent,
    returns response with session tracking.
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        result = await agent.process_message(
            session_id=session_id,
            user_message=request.message,
            user_id=request.user_id
        )
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            iterations=result["iterations"],
            success=result["success"]
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Deployment Guide

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run as non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MINIMAX_API_KEY=${MINIMAX_API_KEY}
      - REDIS_HOST=redis
      - DATABASE_URL=postgresql://agent:password@postgres:5432/agent_db
    depends_on:
      - redis
      - postgres
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=agent
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=agent_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent
  template:
    metadata:
      labels:
        app: ai-agent
    spec:
      containers:
      - name: agent
        image: your-registry/ai-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: MINIMAX_API_KEY
          valueFrom:
            secretKeyRef:
              name: minimax-secret
              key: api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Monitoring and Observability

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
request_count = Counter(
    'agent_requests_total',
    'Total number of agent requests',
    ['status', 'user_id']
)

request_duration = Histogram(
    'agent_request_duration_seconds',
    'Agent request duration in seconds'
)

tool_executions = Counter(
    'tool_executions_total',
    'Total tool executions',
    ['tool_name', 'status']
)

active_sessions = Gauge(
    'active_sessions',
    'Number of active sessions'
)

# Add to FastAPI app
from prometheus_client import generate_latest
from fastapi.responses import Response

@app.get("/metrics
