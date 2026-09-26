---
title: System Design Interview Questions (100)
sidebar_position: 38
sidebar_label: "Reference · System Design"
---

# System Design Interview Questions (100)

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. Start with the [practical topic bank](09-system-design.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](99-tools-versions.md) for current behaviour and [coding labs](11-coding-labs.md) for runnable implementations.
:::

## Core System Design Concepts

<details>
<summary><strong>1. What is system design and why is it important for ML engineers?</strong></summary>

**Answer:**
System design translates ML models to production systems serving millions of users reliably.

**ML Pipeline Components**:
```
[Data Sources] → [Storage] → [Processing] → [Training] 
    ↓                                            ↓
[Monitoring] ← [Serving] ← [Model Registry] ← [Evaluation]
```

**Key aspects**:
1. **Data ingestion**: Collect and store data
2. **Data processing**: ETL, feature engineering
3. **Model training**: Develop and train models
4. **Model serving**: Deploy and inference at scale
5. **Monitoring**: Track performance, detect drift
6. **Feedback loop**: Retrain with new data

```python
# Example ML system architecture
class MLPipeline:
    def __init__(self):
        self.data_store = DataWarehouse()
        self.feature_store = FeatureStore()
        self.model_registry = ModelRegistry()
        self.prediction_service = PredictionService()
        self.monitoring = Monitoring()
    
    def ingest_data(self, data):
        self.data_store.save(data)
    
    def train_model(self):
        features = self.feature_store.get_features()
        model = train(features)
        self.model_registry.register(model)
    
    def serve_predictions(self, input_data):
        model = self.model_registry.get_latest()
        prediction = model.predict(input_data)
        self.monitoring.log(prediction)
        return prediction
```

**Interview Tip**: Discuss end-to-end ML systems, not just models.
</details>

<details>
<summary><strong>2. What is scalability?</strong></summary>

**Answer:**
System's ability to handle growing load.

**Types**:
- **Vertical scaling**: More powerful hardware (CPU, RAM, disk)
- **Horizontal scaling**: More machines (distributed systems)

```python
# Vertical scaling: Single machine gets bigger
# Pros: Simple, no code changes
# Cons: Hardware limits, single point of failure

# Horizontal scaling: Add more machines
# Pros: Unlimited scalability, fault tolerance
# Cons: Distributed systems complexity, consistency issues

# Horizontal scaling pattern for ML:
# - Load balancer distributes requests
# - Multiple prediction servers
# - Shared model storage (NFS, S3)
# - Caching layer (Redis, Memcached)

# Example: Scaling prediction service
class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0
    
    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

class PredictionService:
    def __init__(self, load_balancer):
        self.lb = load_balancer
    
    def predict(self, input_data):
        server = self.lb.get_server()
        return server.predict(input_data)
```

**Interview Tip**: Discuss vertical vs horizontal; know trade-offs.
</details>

<details>
<summary><strong>3. What is availability and reliability?</strong></summary>

**Answer:**
How well system works and doesn't fail.

```
Availability = Uptime / (Uptime + Downtime)

Three 9s (99.9%) = 43 minutes downtime/month
Four 9s (99.99%) = 4 minutes downtime/month
Five 9s (99.999%) = 26 seconds downtime/month
```

**Strategies**:

```python
# 1. Replication (redundancy)
class ReplicatedService:
    def __init__(self, primary, replica):
        self.primary = primary
        self.replica = replica
    
    def request(self, data):
        try:
            return self.primary.process(data)
        except Exception:
            return self.replica.process(data)

# 2. Health checks
class ServiceMesh:
    def __init__(self, services):
        self.services = services
        self.healthy = set(services)
    
    def health_check(self):
        for service in self.services:
            if not service.is_healthy():
                self.healthy.remove(service)

# 3. Circuit breaker (fail fast)
class CircuitBreaker:
    def __init__(self, threshold=5):
        self.failure_count = 0
        self.threshold = threshold
        self.open = False
    
    def call(self, service, request):
        if self.open:
            raise Exception("Circuit breaker is open")
        
        try:
            result = service(request)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.threshold:
                self.open = True
            raise

# 4. Graceful degradation
class DegradedService:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
    
    def serve(self, request):
        try:
            return self.primary.process(request)
        except Exception:
            # Serve fallback (e.g., cached response)
            return self.fallback.get()
```

**Interview Tip**: Know replication, health checks, circuit breakers.
</details>

<details>
<summary><strong>4. What is partitioning/sharding?</strong></summary>

**Answer:**
Splitting data across multiple servers.

```python
# Sharding strategies:

# 1. Range-based sharding
class RangeShard:
    def get_shard(self, key):
        # Users 1-1M → Shard 1
        # Users 1M-2M → Shard 2
        if 1 <= key <= 1000000:
            return "shard_1"
        elif 1000001 <= key <= 2000000:
            return "shard_2"

# 2. Hash-based sharding
class HashShard:
    def __init__(self, num_shards):
        self.num_shards = num_shards
    
    def get_shard(self, key):
        return hash(key) % self.num_shards

# 3. Consistent hashing (for better rebalancing)
class ConsistentHash:
    def __init__(self, nodes=None, replicas=3):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def add_node(self, node):
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            hash_key = hash(virtual_key)
            self.ring[hash_key] = node
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key):
        hash_key = hash(key)
        for ring_key in self.sorted_keys:
            if hash_key <= ring_key:
                return self.ring[ring_key]
        return self.ring[self.sorted_keys[0]]

# Trade-offs:
# Range: Simple but uneven distribution
# Hash: Even distribution but hard to rebalance
# Consistent hash: Even distribution AND easy rebalancing
```

**Interview Tip**: Know consistent hashing for distributed systems.
</details>

<details>
<summary><strong>5. What is caching?</strong></summary>

**Answer:**
Storing frequently accessed data for faster retrieval.

```python
# Cache levels:
# L1: Application cache (in-memory)
# L2: Distributed cache (Redis, Memcached)
# L3: Database
# L4: Persistent storage (S3, disk)

# Cache strategies:

# 1. Write-through (safe but slow)
class WriteThrough:
    def write(self, key, value):
        self.cache[key] = value
        self.database.write(key, value)  # Write to DB first

# 2. Write-behind (fast but risky)
class WriteBehind:
    def write(self, key, value):
        self.cache[key] = value
        # Async write to DB later

# 3. Write-around (for large writes)
class WriteAround:
    def write(self, key, value):
        self.database.write(key, value)  # Skip cache

# Cache eviction policies:

# LRU (Least Recently Used)
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# LFU (Least Frequently Used)
# FIFO (First In First Out)
# Random eviction
```

**Interview Tip**: Know cache strategies and eviction policies.
</details>

<details>
<summary><strong>6. What is load balancing?</strong></summary>

**Answer:**
Distributing requests across multiple servers.

```python
# Load balancing algorithms:

# 1. Round Robin
class RoundRobinLB:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0
    
    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# 2. Least Connections
class LeastConnectionLB:
    def __init__(self, servers):
        self.servers = servers
        self.connections = {s: 0 for s in servers}
    
    def get_server(self):
        return min(self.servers, key=lambda s: self.connections[s])

# 3. Weighted Round Robin
class WeightedRRLB:
    def __init__(self, servers, weights):
        self.servers = servers
        self.weights = weights
        self.current = 0
    
    def get_server(self):
        # Servers with higher weight get more traffic
        pass

# 4. IP Hash (for session affinity)
class IPHashLB:
    def __init__(self, servers):
        self.servers = servers
    
    def get_server(self, client_ip):
        index = hash(client_ip) % len(self.servers)
        return self.servers[index]

# 5. Least Response Time
class ResponseTimeLB:
    def get_server(self):
        return min(self.servers, 
                   key=lambda s: s.get_average_response_time())
```

**Interview Tip**: Know different algorithms and when to use each.
</details>

<details>
<summary><strong>7. What is database design?</strong></summary>

**Answer:**
Structuring data for efficiency and correctness.

```python
# Relational database example:
"""
Users table:
- user_id (PK)
- name
- email

Orders table:
- order_id (PK)
- user_id (FK)
- amount
- date

SQL:
SELECT u.name, COUNT(o.order_id) as order_count
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id
"""

# NoSQL (document) database:
"""
users_collection:
{
    "_id": "user_123",
    "name": "Alice",
    "email": "alice@example.com",
    "orders": [
        {"order_id": "ord_1", "amount": 100},
        {"order_id": "ord_2", "amount": 200}
    ]
}
"""

# Normalization (relational):
# Reduces redundancy, maintains integrity
# 1NF, 2NF, 3NF, BCNF

# Denormalization (NoSQL):
# Improves read performance, controlled redundancy

# Key design decisions:
# 1. SQL vs NoSQL
# 2. Normalization vs Denormalization
# 3. Indexing strategy
# 4. Sharding key choice
```

**Interview Tip**: Discuss SQL vs NoSQL trade-offs for use case.
</details>

<details>
<summary><strong>8. What is API design?</strong></summary>

**Answer:**
Interface for services to communicate.

```python
# RESTful API principles

# 1. Resource-based URLs
GET    /users              # List users
POST   /users              # Create user
GET    /users/{id}         # Get specific user
PUT    /users/{id}         # Update user
DELETE /users/{id}         # Delete user

# 2. Status codes
200 OK                     # Successful request
201 Created               # Resource created
400 Bad Request           # Invalid input
401 Unauthorized          # Authentication failed
403 Forbidden             # Not authorized
404 Not Found             # Resource not found
500 Server Error          # Unexpected error

# 3. Request/Response format
{
    "user_id": 123,
    "name": "Alice",
    "email": "alice@example.com"
}

# 4. API versioning
/api/v1/users
/api/v2/users  # Breaking changes in v2

# 5. Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.limit = requests_per_minute
        self.requests = {}
    
    def is_allowed(self, client_id):
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < 60
        ]
        
        if len(self.requests[client_id]) < self.limit:
            self.requests[client_id].append(now)
            return True
        return False
```

**Interview Tip**: Design clear, consistent APIs with proper status codes.
</details>

<details>
<summary><strong>9. What is monitoring and logging?</strong></summary>

**Answer:**
Tracking system health and debugging issues.

```python
# Metrics to monitor
# - Request latency (p50, p95, p99)
# - Error rate
# - CPU, memory, disk usage
# - Database query time
# - Model performance (accuracy, precision, recall)

class Monitor:
    def __init__(self):
        self.metrics = {}
    
    def record_latency(self, endpoint, latency):
        if endpoint not in self.metrics:
            self.metrics[endpoint] = []
        self.metrics[endpoint].append(latency)
    
    def get_percentile(self, endpoint, p):
        values = sorted(self.metrics[endpoint])
        index = int(len(values) * p / 100)
        return values[index]

# Logging levels
# DEBUG: Detailed info for debugging
# INFO: General informational messages
# WARNING: Something unexpected
# ERROR: Error that needs attention
# CRITICAL: System failure

import logging

logger = logging.getLogger(__name__)
logger.debug("Processing request")
logger.info("Model loaded successfully")
logger.warning("Cache miss rate is high")
logger.error("Failed to connect to database")

# Distributed tracing
class TraceContext:
    def __init__(self, trace_id):
        self.trace_id = trace_id
        self.spans = []
    
    def add_span(self, span_name, duration):
        self.spans.append({
            'name': span_name,
            'duration': duration
        })
```

**Interview Tip**: Know monitoring, logging, and how to debug production issues.
</details>

<details>
<summary><strong>10. What is security?</strong></summary>

**Answer:**
Protecting system from unauthorized access and attacks.

```python
# Authentication (who you are)
class Authentication:
    def login(self, username, password):
        user = db.get_user(username)
        if not user or not verify_password(password, user.password_hash):
            raise AuthenticationError()
        
        # Generate token
        token = jwt.encode({'user_id': user.id}, SECRET_KEY)
        return token

# Authorization (what you can do)
class Authorization:
    def is_allowed(self, user_id, resource):
        permissions = db.get_permissions(user_id)
        return resource in permissions

# Encryption
import hashlib
from cryptography.fernet import Fernet

# Hash passwords
password_hash = hashlib.pbkdf2_hmac(
    'sha256',
    password.encode(),
    salt,
    100000
)

# Encrypt sensitive data
cipher = Fernet(key)
encrypted = cipher.encrypt(sensitive_data)

# HTTPS/TLS
# - Encrypt data in transit
# - Verify server identity
# - Prevent man-in-the-middle attacks

# SQL injection prevention
# DON'T: f"SELECT * FROM users WHERE id = {user_id}"
# DO: db.execute("SELECT * FROM users WHERE id = ?", [user_id])

# API security
# - Rate limiting (prevent brute force)
# - Input validation (prevent injection)
# - CORS (control who can access API)
# - HTTPS (encrypt transit)
```

**Interview Tip**: Know authentication, authorization, encryption basics.
</details>

---

## Distributed Systems

<details>
<summary><strong>11. How do message queues work and when should you use them?</strong></summary>

**Answer:**
Message queues decouple producers from consumers, enabling async processing and buffering traffic spikes.

```
Producer → [Queue: Kafka/RabbitMQ/SQS] → Consumer(s)
```

```python
from confluent_kafka import Producer, Consumer

producer = Producer({'bootstrap.servers': 'localhost:9092'})

def send_event(topic, event):
    producer.produce(topic, value=str(event).encode())
    producer.flush()

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'ml-pipeline-group',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe(['raw-events'])

while True:
    msg = consumer.poll(timeout=1.0)
    if msg and not msg.error():
        process_event(msg.value())

# Kafka concepts:
# - Partition: unit of parallelism
# - Offset: position within partition
# - Consumer group: scales horizontally
# - Log compaction: keep latest per key

# Use queues for: async ML inference, event streaming,
# decoupling services, load leveling, guaranteed delivery
```

**When NOT to use**: request-response patterns needing ultra-low latency.
</details>

<details>
<summary><strong>12. What is the CAP theorem?</strong></summary>

**Answer:**
In a distributed system you can guarantee only 2 of 3: Consistency, Availability, Partition Tolerance. Since partitions always happen in practice, choose CP or AP.

```
CP systems: consistent, may be unavailable during partition
  → HBase, MongoDB (strong reads), Zookeeper

AP systems: always available, may return stale data
  → Cassandra, DynamoDB, CouchDB

# DynamoDB (AP): always accepts writes, eventual consistency default
# Zookeeper (CP): refuses writes during partition, linearizable reads

# For ML systems:
# Feature store → AP (serve features even if slightly stale)
# Model registry → CP (must have consistent model versions)
# Experiment tracking → CP (accurate metrics critical)
```

**PACELC extension**: even without partition, tradeoff between latency and consistency.
</details>

<details>
<summary><strong>13. How does consistent hashing work?</strong></summary>

**Answer:**
Consistent hashing minimizes remapping when nodes are added/removed — only K/N keys move instead of all K.

```python
import hashlib
import bisect

class ConsistentHashRing:
    def __init__(self, replicas=3):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        for i in range(self.replicas):
            h = self._hash(f"{node}:vn{i}")
            self.ring[h] = node
            bisect.insort(self.sorted_keys, h)

    def remove_node(self, node):
        for i in range(self.replicas):
            h = self._hash(f"{node}:vn{i}")
            del self.ring[h]
            self.sorted_keys.remove(h)

    def get_node(self, key):
        if not self.ring: return None
        h = self._hash(key)
        idx = bisect.bisect(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]

ring = ConsistentHashRing(replicas=3)
for node in ['server1', 'server2', 'server3']:
    ring.add_node(node)

print(ring.get_node("user_123"))  # consistent assignment
# Use cases: distributed caches, sharded DBs, CDN routing
```
</details>

<details>
<summary><strong>14. What is the Saga pattern for distributed transactions?</strong></summary>

**Answer:**
Saga coordinates transactions across services using compensating actions for rollback — no global lock needed.

```python
# Choreography-based Saga (event-driven)
class OrderService:
    def create_order(self, order_id, amount):
        db.create_order(order_id, status='PENDING')
        kafka.publish('order-created', {'order_id': order_id, 'amount': amount})

class PaymentService:
    def on_order_created(self, event):
        try:
            charge_card(event['amount'])
            kafka.publish('payment-succeeded', event)
        except PaymentFailed:
            kafka.publish('payment-failed', event)  # trigger rollback

class OrderService:
    def on_payment_failed(self, event):
        db.update_order(event['order_id'], status='CANCELLED')

# Orchestration-based Saga (central coordinator)
class SagaOrchestrator:
    def execute(self, order_id):
        steps = [
            (payment_service.charge,    payment_service.refund),
            (inventory_service.reserve, inventory_service.release),
            (shipping_service.schedule, shipping_service.cancel),
        ]
        completed = []
        for action, compensate in steps:
            try:
                action(order_id)
                completed.append(compensate)
            except Exception:
                for comp in reversed(completed):
                    comp(order_id)   # compensating transactions
                raise

# Saga = eventual consistency, not ACID
```
</details>

<details>
<summary><strong>15. What is CQRS and Event Sourcing?</strong></summary>

**Answer:**
CQRS separates read/write models; Event Sourcing stores state as a sequence of immutable events.

```python
# CQRS: Command = write, Query = read (separate models)
class CommandHandler:
    def handle_create_user(self, cmd):
        user = User(cmd.user_id, cmd.email)
        user_repository.save(user)
        event_store.append(UserCreatedEvent(cmd.user_id, cmd.email))

class QueryHandler:
    # Read model: denormalized, read-optimized
    def get_user_profile(self, user_id):
        return read_db.query("SELECT * FROM user_profiles WHERE id = ?", user_id)

# Event Sourcing: rebuild state by replaying events
class UserAggregate:
    def rebuild(self, user_id):
        events = event_store.get_events(user_id)
        state = {}
        for event in events:
            if event.type == 'UserCreated':   state.update(event.data)
            elif event.type == 'EmailChanged': state['email'] = event.data['email']
        return state

# Benefits: full audit log, time travel, easy A/B testing
# Drawbacks: eventual consistency, complex queries
```
</details>

## ML System Design

<details>
<summary><strong>16. How do you design a recommendation system at scale?</strong></summary>

**Answer:**
Two-stage retrieval-ranking pipeline: candidate generation (millions → thousands) then ranking (thousands → tens).

```python
import torch
import torch.nn as nn

# Stage 1: Two-Tower for candidate generation
class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, embed_dim=64):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, user_feats, item_feats):
        u = self.user_tower(user_feats)
        i = self.item_tower(item_feats)
        return torch.sigmoid(torch.sum(u * i, dim=-1))

# Serving: pre-compute item embeddings, ANN search at runtime
import faiss
import numpy as np

index = faiss.IndexFlatIP(64)          # inner product similarity
index.add(item_embeddings)             # pre-index all items
D, I = index.search(user_emb, k=100)  # retrieve top-100 candidates

# Stage 2: Ranking model (LightGBM or Wide&Deep)
# Features: user history, item metadata, context, cross-features

# Stage 3: Re-ranking (business rules)
# diversity, freshness boost, sponsored content

# Architecture:
# [User Activity] -> [Feature Store] -> [Two-Tower] -> [FAISS ANN]
#                                                          |
#                                                    [Ranker] -> [Re-ranker] -> [UI]
```

**Key metrics**: CTR, conversion rate, diversity, coverage, serendipity.
</details>

<details>
<summary><strong>17. How do you design a feature store?</strong></summary>

**Answer:**
Feature store unifies feature computation for training and serving with point-in-time correctness.

```python
# Online serving (Redis)
import redis
r = redis.Redis()

def get_online_features(user_id):
    key = f"features:user:{user_id}"
    features = r.hgetall(key)
    return {k.decode(): float(v) for k, v in features.items()}

def set_online_features(user_id, features, ttl=3600):
    r.hset(f"features:user:{user_id}", mapping=features)
    r.expire(f"features:user:{user_id}", ttl)

# Offline: Parquet on S3, partitioned by date
# Training join: feature_timestamp <= label_timestamp (point-in-time!)

# Feature pipeline (scheduled batch job)
def compute_user_features(date):
    return spark.sql(f"""
    SELECT
        user_id,
        AVG(purchase_amount) OVER (
            PARTITION BY user_id
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '7' DAY PRECEDING AND CURRENT ROW
        ) as avg_purchase_7d,
        COUNT(*) OVER (
            PARTITION BY user_id
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
        ) as sessions_30d
    FROM events WHERE DATE(event_time) = '{date}'
    """)

# Key requirements:
# 1. Point-in-time correctness (no label leakage)
# 2. Feature reuse (compute once, use everywhere)
# 3. Low-latency serving (< 10ms p99)
# 4. Backfilling (compute historical features for training)
```

**Tools**: Feast, Tecton, Hopsworks, Vertex AI Feature Store.
</details>

<details>
<summary><strong>18. How do you design an A/B testing framework?</strong></summary>

**Answer:**
A/B testing needs traffic splitting, consistent user assignment, statistical significance testing, and guardrail metrics.

```python
import hashlib
from scipy import stats
import numpy as np

class ABTestFramework:
    def __init__(self):
        self.experiments = {}

    def create_experiment(self, exp_id, variants, traffic_pct=0.1):
        self.experiments[exp_id] = {
            'variants': variants,
            'traffic_pct': traffic_pct,
        }

    def get_variant(self, exp_id, user_id):
        exp = self.experiments[exp_id]
        # Hash for consistent assignment (same user, same variant)
        bucket = int(hashlib.md5(f"{exp_id}:{user_id}".encode()).hexdigest(), 16) % 100
        if bucket >= exp['traffic_pct'] * 100:
            return None   # not in experiment
        return exp['variants'][bucket % len(exp['variants'])]

    def analyze(self, control, treatment, alpha=0.05):
        t_stat, p_value = stats.ttest_ind(control, treatment)
        effect = (np.mean(treatment) - np.mean(control)) / np.std(control)
        return {
            'p_value': round(p_value, 4),
            'significant': p_value < alpha,
            'effect_size': round(effect, 3),
            'relative_lift_pct': round((np.mean(treatment)/np.mean(control) - 1)*100, 2)
        }

# Guardrail metrics: never degrade (latency, error rate)
# Primary metrics: improve (CTR, conversion rate)
# Network effects: if users interact, use switchback tests (not user-level split)
# Minimum detectable effect: determine sample size before launch
```
</details>

<details>
<summary><strong>19. How do you design model serving infrastructure?</strong></summary>

**Answer:**
Model serving needs versioned endpoints, auto-scaling, dynamic batching, and monitoring.

```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = torch.jit.load("model.pt")
    model.eval()

@app.post("/predict")
async def predict(request: PredictRequest):
    with torch.no_grad():
        features = torch.tensor(request.features)
        output = model(features)
    return {"prediction": output.tolist()}

# Dynamic batching: collect requests, run one GPU forward pass
import asyncio

class BatchPredictor:
    def __init__(self, model, batch_size=32, timeout_ms=10):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000
        self.queue = asyncio.Queue()

    async def add_request(self, features):
        future = asyncio.Future()
        await self.queue.put((features, future))
        return await future

    async def process_batches(self):
        while True:
            batch, futures = [], []
            try:
                while len(batch) < self.batch_size:
                    feat, fut = await asyncio.wait_for(self.queue.get(), self.timeout)
                    batch.append(feat); futures.append(fut)
            except asyncio.TimeoutError:
                pass
            if batch:
                results = self.model(torch.stack(batch))
                for fut, res in zip(futures, results):
                    fut.set_result(res)

# Deployment strategies:
# Canary: route 5% traffic to new model, monitor, ramp up
# Shadow: run new model in parallel (no user impact), compare outputs
# Blue-green: two envs, instant traffic switch
```
</details>

<details>
<summary><strong>20. How do you design a data lake and data warehouse?</strong></summary>

**Answer:**
Data lake stores raw data cheaply (S3/GCS); data warehouse stores structured data for analytics. Modern lakehouse (Delta Lake, Iceberg) combines both.

```python
# Delta Lake: ACID transactions on S3
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0") \
    .getOrCreate()

# Upsert (merge) with ACID guarantees
delta_table = DeltaTable.forPath(spark, "s3://bucket/users")
new_data = spark.createDataFrame([...])

delta_table.alias("old").merge(
    new_data.alias("new"),
    "old.user_id = new.user_id"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

# Time travel: read any historical version
df_yesterday = spark.read.format("delta") \
    .option("versionAsOf", 10) \
    .load("s3://bucket/users")

# Data zones:
# Raw: as-is from sources (Parquet/Avro)
# Curated: cleaned, validated, deduplicated
# Serving: aggregated, feature store, data marts

# Partitioning strategy:
# Events: partition by event_date (fast time-range queries)
# Users: no partition (small table, full scan OK)
# Logs: partition by date + hour
```

**Tools**: dbt (SQL transforms), Spark (large-scale ETL), Great Expectations (data quality).
</details>

<details>
<summary><strong>21. Rate Limiting Algorithms</strong></summary>

```python

# Token Bucket: tokens accumulate at fixed rate, consumed per request
# Leaky Bucket: requests drain at fixed rate (smoothing bursts)
# Sliding Window Counter: approximate, memory efficient

# Redis sliding window rate limiter
import time
import redis
r = redis.Redis()

def is_allowed(user_id, rate=100, window_sec=1):
    key = f"ratelimit:{user_id}"
    now = time.time()
    pipe = r.pipeline()
    pipe.zadd(key, {now: now})
    pipe.zremrangebyscore(key, 0, now - window_sec)
    pipe.zcard(key)
    pipe.expire(key, 2)
    _, _, count, _ = pipe.execute()
    return count <= rate
```
</details>

<details>
<summary><strong>22. Service Mesh and Microservices</strong></summary>

```

Service Mesh (Istio/Linkerd):
- Sidecar proxies: load balancing, mTLS, retries, circuit breaking, tracing
- Control plane: configures all proxies

Microservice patterns:
- API Gateway: single entry, auth, rate limit, routing
- BFF (Backend for Frontend): tailored APIs per client type
- Strangler Fig: gradually replace monolith
```
</details>

<details>
<summary><strong>23. Database Sharding Strategies</strong></summary>

```

Range-based: shard by ID range (simple, hot-spot risk for sequential IDs)
Hash-based: hash(key) % N (even distribution, hard to rebalance)
Directory-based: lookup table maps key to shard (flexible, SPOF)
Geo-based: shard by region (compliance, latency)

Cross-shard queries require scatter-gather or denormalization.
```
</details>

<details>
<summary><strong>24. Caching Patterns</strong></summary>

```python

# Cache-aside (lazy loading)
def get_user(user_id):
    cached = redis.get(f"user:{user_id}")
    if cached: return json.loads(cached)
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    redis.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user

# Write-through: write to cache AND DB simultaneously (strong consistency)
# Write-behind: write cache first, async DB (risk data loss on crash)
# Cache stampede: many misses simultaneously -> use locks or early expiry jitter
```
</details>

<details>
<summary><strong>25. CDN and Edge Computing</strong></summary>

```

CDN (CloudFront/Fastly/Akamai):
- Cache static assets at PoPs (Points of Presence) globally
- Reduce latency for global users (ms vs seconds)
- DDoS protection, SSL termination

Edge ML: run lightweight models at edge
- Lower latency (no round-trip to cloud)
- Privacy (data stays on device)
- Runtimes: ONNX, TFLite, CoreML
```
</details>

<details>
<summary><strong>26. Kubernetes Resource Management</strong></summary>

```yaml

apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: ml-server
        image: ml-model:v2
        resources:
          requests: {cpu: "500m", memory: "1Gi"}
          limits:   {cpu: "2",    memory: "4Gi"}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource: {name: cpu, target: {averageUtilization: 70}}
```
</details>

<details>
<summary><strong>27. ML Monitoring and Drift Detection</strong></summary>

```python

from scipy.stats import ks_2samp
import numpy as np

def detect_drift(reference, production, threshold=0.05):
    stat, p_value = ks_2samp(reference, production)
    return {'drifted': p_value < threshold, 'p_value': round(p_value, 4)}

def psi(expected, actual, buckets=10):
    bins = np.percentile(expected, np.linspace(0, 100, buckets+1))
    e_pct = np.histogram(expected, bins)[0] / len(expected) + 1e-6
    a_pct = np.histogram(actual,   bins)[0] / len(actual)   + 1e-6
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    # PSI < 0.1: stable; 0.1-0.25: moderate shift; > 0.25: major shift
```
</details>

<details>
<summary><strong>28. Designing a Search Engine</strong></summary>

```

Components:
1. Crawler: fetch pages, respect robots.txt
2. Parser: extract text, links, metadata
3. Inverted Index: term -> [(doc_id, positions)]
4. Ranking: TF-IDF, BM25, PageRank, learned ranking (LTR)
5. Query processing: tokenize, spell-correct, synonyms

Elasticsearch scaling:
- Shards: horizontal split of index
- Replicas: copies for fault tolerance + read scaling
- Segment merging: compact small segments periodically
```
</details>

<details>
<summary><strong>29. Real-Time ML Pipeline Design</strong></summary>

```

[Kafka] -> [Flink] -> [Feature Engineering] -> [Model Inference] -> [Result DB]
                              |
                       [Redis feature cache]

Latency budget (p99 < 100ms):
- Feature lookup:  5ms (Redis)
- Model inference: 20ms (GPU batch)
- Post-processing:  5ms
- Network:         20ms
- Total:           50ms (50ms headroom)
```
</details>

<details>
<summary><strong>30. Distributed Training Strategies</strong></summary>

```

Data Parallelism: same model, different data batches per GPU
  - AllReduce gradient sync (NCCL)
  - PyTorch DDP, Horovod

Model Parallelism: split model across GPUs (for huge models)
  - Pipeline parallelism: split by layer
  - Tensor parallelism: split weight matrices

ZeRO (DeepSpeed):
  - ZeRO-1: shard optimizer states
  - ZeRO-2: + gradients
  - ZeRO-3: + model parameters (enables 100B+ models)
```
</details>

<details>
<summary><strong>31. Vector Databases for ML</strong></summary>

```python

import faiss
import numpy as np

d = 128
index = faiss.IndexFlatIP(d)       # exact inner product
# Approximate: IndexIVFFlat, IndexHNSW (faster, slight quality loss)

embeddings = np.random.rand(10000, d).astype('float32')
index.add(embeddings)
D, I = index.search(query_vec, k=10)

# Managed options: Pinecone, Weaviate, Qdrant, Milvus
# Hybrid search (vector + keyword): Reciprocal Rank Fusion
def rrf_merge(vec_results, kw_results, k=60):
    scores = {}
    for rank, doc_id in enumerate(vec_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1/(k + rank)
    for rank, doc_id in enumerate(kw_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1/(k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```
</details>

<details>
<summary><strong>32. LLM Serving Infrastructure</strong></summary>

```

Challenges:
- KV cache: stores attention keys/values (large GPU memory)
- Variable sequence lengths: inefficient batching
- Streaming: token-by-token output

vLLM: PagedAttention (paged KV cache, 2-4x throughput)
Triton Inference Server: multi-model, concurrent execution
TensorRT-LLM: NVIDIA optimized inference

Optimization techniques:
- FlashAttention: memory-efficient attention
- Quantization: GPTQ/AWQ (4-bit, 4x smaller)
- Speculative decoding: draft model + verifier
- Tensor parallelism: split model across GPUs
```
</details>

<details>
<summary><strong>33. Observability: Metrics, Logs, Traces</strong></summary>

```python

from prometheus_client import Counter, Histogram, Gauge

request_count   = Counter('requests_total', 'Total', ['method', 'endpoint'])
request_latency = Histogram('request_latency_seconds', 'Latency')
active_models   = Gauge('active_model_count', 'Models loaded')

@request_latency.time()
def handle(method, endpoint):
    request_count.labels(method=method, endpoint=endpoint).inc()

# Distributed tracing (OpenTelemetry)
from opentelemetry import trace
tracer = trace.get_tracer("ml-service")
with tracer.start_as_current_span("predict") as span:
    span.set_attribute("model.version", "v2.1")
    result = model.predict(features)

# Structured logging
import structlog
log = structlog.get_logger()
log.info("prediction_made", user_id=uid, latency_ms=50, model="v2.1")
```
</details>

<details>
<summary><strong>34. Multi-Region Deployment</strong></summary>

```

Active-Active: multiple regions serve traffic, sync via replication
- Lower latency globally, complex conflict resolution for writes

Active-Passive: primary region + standby regions
- Simpler, failover takes time (RTO), replication lag (RPO)

Data residency: GDPR requires EU data stays in EU
- Partition user data by region; metadata stored globally
```
</details>

<details>
<summary><strong>35. Disaster Recovery Planning</strong></summary>

```

Key metrics:
- RTO (Recovery Time Objective): max acceptable downtime
- RPO (Recovery Point Objective): max acceptable data loss

Strategies (cheapest to most expensive):
1. Backup & restore: hours RTO/RPO
2. Pilot light: minimal infra always on, scale on failure (minutes)
3. Warm standby: scaled-down running copy (minutes)
4. Multi-site active-active: zero downtime, highest cost

WAL shipping: stream DB transaction logs to replica continuously
Snapshot: filesystem-level point-in-time copy
```
</details>

<details>
<summary><strong>36. Online Learning Systems</strong></summary>

```python

# Continuously update model with new data
# Use cases: fraud detection, ad CTR, real-time personalization
from river import linear_model, optim, preprocessing, metrics

model = (preprocessing.StandardScaler() |
         linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.01)))
metric = metrics.ROCAUC()

for features, label in data_stream:
    pred = model.predict_proba_one(features)
    metric.update(label, pred)
    model.learn_one(features, label)

# Challenges:
# - Catastrophic forgetting
# - Concept drift detection
# - Training-serving skew
```
</details>

<details>
<summary><strong>37. Privacy-Preserving ML</strong></summary>

```

Federated Learning: train on device, aggregate gradients (not data)
  Used by Google (Gboard), Apple (Siri)

Differential Privacy: add calibrated noise to protect individuals
  epsilon-DP: smaller = more private
  Used by Apple, US Census Bureau

DP-SGD: gradient clipping + noise addition during training
```
</details>

<details>
<summary><strong>38. Designing Twitter Trending Topics</strong></summary>

```

Problem: compute trending topics in real-time from millions of tweets

1. Kafka: ingest tweet stream (58K tweets/sec at 10x peak)
2. Flink: count hashtags in sliding 1-hour window
3. Velocity scoring: current_count / expected_count
4. Redis Sorted Set: ZADD trending {score} {hashtag}
5. API: ZREVRANGE trending 0 9 -> top 10

Personalization: trending in your network vs globally
Geography: partition by region (country/city)
Bot filtering: deduplicate by IP + device fingerprint
```
</details>

<details>
<summary><strong>39. Handling Class Imbalance in Production</strong></summary>

```python

from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
import numpy as np

# SMOTE: synthesize minority class samples
X_res, y_res = SMOTE(sampling_strategy=0.1).fit_resample(X_train, y_train)

# Cost-sensitive learning
model = RandomForestClassifier(class_weight={0: 1, 1: 100})

# Threshold tuning (don't default to 0.5)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
# Choose threshold that maximizes F-beta or meets business constraint

# Evaluation: use PR-AUC, not ROC-AUC (insensitive to imbalance)
```
</details>

<details>
<summary><strong>40. MLOps: CI/CD for Machine Learning</strong></summary>

```yaml

# GitHub Actions ML pipeline
name: ML CI/CD
on: [push]
jobs:
  test:
    steps:
    - run: pytest tests/unit/ tests/model_tests.py
  train:
    needs: test
    steps:
    - run: python train.py --config config.yaml
    - run: python evaluate.py --threshold 0.95
    - run: mlflow models register -m runs:/$RUN_ID/model -n MyModel
  deploy:
    needs: train
    steps:
    - run: kubectl set image deployment/ml-server container=model:$VERSION
    - run: kubectl rollout status deployment/ml-server
    - run: python smoke_test.py --endpoint $ENDPOINT
```
</details>

<details>
<summary><strong>41. Data Versioning and Lineage</strong></summary>

```python

# DVC: Git for data and models
# dvc add data/train.csv  -> tracked in .dvc file
# dvc push               -> upload to S3/GCS
# dvc checkout           -> restore exact version

import mlflow
with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("val_auc", 0.95)
    mlflow.log_artifact("model.pkl")
    mlflow.set_tag("data_version", "v2024-01-15")

# Lineage tools: Apache Atlas, DataHub, Amundsen, OpenLineage
```
</details>

<details>
<summary><strong>42. Cost Optimization for ML</strong></summary>

```

Training:
- Spot/preemptible instances: 60-90% cheaper, can be preempted
- Mixed precision (FP16/BF16): 2x memory savings, 2-3x speed
- Gradient checkpointing: trade compute for memory

Serving:
- Quantization (INT8): 4x smaller, 2-4x faster
- Pruning: remove low-importance weights
- Dynamic batching: amortize GPU overhead across requests
- Rightsizing: don't over-provision GPUs

Data:
- Tiered storage: hot (SSD) -> warm (HDD) -> cold (Glacier)
- Compression: Parquet + Snappy (5-10x reduction)
```
</details>

<details>
<summary><strong>43. Designing a Fraud Detection System</strong></summary>

```

Requirements: real-time (<100ms), high recall, low false positives

Architecture:
[Transaction] -> [Rule Engine] -> [ML Scorer] -> [Decision] -> [Case Queue]

Features:
- Velocity: N transactions last hour/day
- Merchant risk: historical fraud rate
- Geo-mismatch: IP location vs billing address
- Device fingerprint: new device for account
- Behavioral: unusual time, amount, category

Models:
- LightGBM: fast, interpretable
- GNN: detect collusion rings
- Autoencoder: unsupervised anomaly detection

Challenge: adversarial adaptation (fraudsters adjust to detection)
```
</details>

<details>
<summary><strong>44. AutoML System Design</strong></summary>

```

HPO: Bayesian optimization, Hyperband, ASHA (async successive halving)
NAS: search over architectures (DARTS, one-shot)
Feature selection: importance + recursive elimination
Model selection: train multiple, pick best on hold-out

Tools: Google AutoML, H2O, AutoGluon, TPOT, Ludwig, Ray Tune
```
</details>

<details>
<summary><strong>45. GraphQL vs REST vs gRPC</strong></summary>

```

REST: simple, cacheable, widely supported
  - Over/under-fetching, multiple round-trips

GraphQL: flexible queries, single endpoint, strong typing
  - Harder caching, n+1 problem, complexity

gRPC: binary (Protobuf), streaming, auto-generated clients
  - 2-10x faster than REST for internal services
  - Poor browser support

Rule: REST for public APIs, gRPC for internal services,
GraphQL for flexible frontend contracts
```
</details>

<details>
<summary><strong>46. Service Discovery and Load Balancing</strong></summary>

```

Client-side: client queries registry (Consul/Eureka), picks server itself
Server-side: client -> LB -> registry -> server

Load balancing algorithms:
- Round-robin: even distribution, ignores actual load
- Least connections: route to server with fewest active connections
- Consistent hashing: session affinity (stateful services)
- Power of 2 random: pick 2 random, route to less loaded (better than RR)
```
</details>

<details>
<summary><strong>47. Data Streaming: Kafka vs Kinesis</strong></summary>

```

Kafka:
+ High throughput, log retention, consumer groups, exactly-once
- Ops overhead (ZooKeeper/KRaft cluster management)

Kinesis:
+ Managed AWS, auto-scaling
- 24h retention (max 7 days), limited per-shard throughput

Pub/Sub (GCP):
+ Globally distributed, auto-scaling, push/pull
- No ordering guarantees across subscriptions
```
</details>

<details>
<summary><strong>48. Distributed Locking</strong></summary>

```python

import redis, uuid, time
r = redis.Redis()

def acquire_lock(name, timeout=10):
    lock_id = str(uuid.uuid4())
    if r.set(name, lock_id, nx=True, ex=timeout):
        return lock_id
    return None

def release_lock(name, lock_id):
    if r.get(name) == lock_id.encode():
        r.delete(name)
    # Redlock: acquire on N/2+1 Redis instances for stronger guarantees
```
</details>

<details>
<summary><strong>49. Multi-Tenancy in ML Platforms</strong></summary>

```

Isolation levels:
- Kubernetes namespaces + resource quotas
- Separate storage buckets/DB schemas per tenant
- Separate serving endpoints or model tags
- Billing: track usage per tenant with labels

Row-level security: filter data by tenant_id at query time
Tenant metadata: global service maps tenant_id to configs
```
</details>

<details>
<summary><strong>50. SLOs, SLAs, Error Budgets</strong></summary>

```

SLI: measurable indicator (p99 latency = 95ms)
SLO: target (p99 latency < 100ms, 99.9% of time)
SLA: contractual SLO with financial penalties

Error budget = 100% - SLO target
99.9% SLO = 0.1% budget = 43.8 min/month downtime allowed

When budget is exhausted: freeze feature deployments, focus on reliability
When budget is healthy: invest in feature velocity
```
</details>

<details>
<summary><strong>51. Database Indexing Deep Dive</strong></summary>

```sql

-- Composite index: leftmost prefix rule
CREATE INDEX idx_status_date ON users(status, created_at);
-- Good: WHERE status='active' AND created_at > '2024-01-01'
-- Bad (ignores index): WHERE created_at > '2024-01-01' only

-- Covering index: includes all query columns
CREATE INDEX idx_covering ON orders(user_id, status, amount);
-- Query: SELECT user_id, status, amount FROM orders WHERE user_id=1
-- -> index-only scan (no heap fetch)

-- Partial index: index subset of rows
CREATE INDEX idx_active ON users(email) WHERE status = 'active';
```
</details>

<details>
<summary><strong>52. Hot Partition Handling</strong></summary>

```

Cause: one shard gets disproportionate traffic (e.g., celebrity account)
Solutions:
- Random suffix on key: shard_key + random(0,N) -> write amplification
- Micro-partitioning: more shards than nodes
- Caching: absorb hot reads before hitting shard
- Push model: pre-fan-out to follower shards for writes
- Hybrid: push for regular users, pull for celebrities (Twitter)
```
</details>

<details>
<summary><strong>53. Backpressure</strong></summary>

```

When producer is faster than consumer:
- Bounded queues: producer blocks when queue full
- Reactive Streams: subscriber signals demand upstream
- Drop/sample: acceptable for metrics, not transactions
- Circuit breaker: stop accepting new requests

Flink: watermarks + timer service manage backpressure in streaming
```
</details>

<details>
<summary><strong>54. Schema Evolution</strong></summary>

```

Backward compatible: new fields with defaults (old readers work with new data)
Forward compatible: old readers ignore unknown fields
Full compatible: both

Avro/Protobuf/Thrift: designed for schema evolution
Database expand-contract: add nullable column -> migrate -> drop old column
Schema Registry: enforces compatibility before publish
```
</details>

<details>
<summary><strong>55. Multi-Model Serving (Ensembling)</strong></summary>

```python

class EnsemblePredictor:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)

    def predict(self, X):
        import numpy as np
        preds = [m.predict_proba(X) for m in self.models]
        return np.average(preds, axis=0, weights=self.weights)

# Triton: run multiple models concurrently on one GPU server
# Model multiplexing: one server handles many model versions
```
</details>

<details>
<summary><strong>56. Shadow Mode for Safe Model Rollout</strong></summary>

```python

import threading

class ShadowPredictor:
    def __init__(self, primary, shadow):
        self.primary = primary
        self.shadow = shadow

    def predict(self, features):
        result = self.primary.predict(features)
        # Shadow call in background (non-blocking)
        def shadow_call():
            try:
                shadow_result = self.shadow.predict(features)
                log_comparison(result, shadow_result)
            except Exception as e:
                log_error(e)
        threading.Thread(target=shadow_call, daemon=True).start()
        return result  # only serve primary result
```
</details>

<details>
<summary><strong>57. Database Connection Pooling</strong></summary>

```python

from sqlalchemy import create_engine
engine = create_engine(
    "postgresql://user:pass@host/db",
    pool_size=20,       # persistent connections
    max_overflow=10,    # temporary extras
    pool_timeout=30,    # wait time
    pool_recycle=3600   # recycle hourly
)
# PgBouncer: transaction-mode pooling
# 10K app connections -> 100 DB connections
```
</details>

<details>
<summary><strong>58. OLAP vs OLTP</strong></summary>

```

OLTP: many short transactions, row-oriented, normalized, write-heavy
  -> MySQL, Postgres, MongoDB

OLAP: complex aggregations, column-oriented, denormalized, read-heavy
  -> BigQuery, Redshift, ClickHouse, Snowflake

HTAP: both (TiDB, SingleStore, CockroachDB)
```
</details>

<details>
<summary><strong>59. Chaos Engineering</strong></summary>

```

Principles:
1. Define steady state (normal metrics baseline)
2. Hypothesize: "system will stay stable when X fails"
3. Inject failure (kill instance, add latency, drop packets)
4. Verify steady state maintained

Tools: Netflix Chaos Monkey, Gremlin, Chaos Mesh (Kubernetes)
Game days: planned chaos experiments with team participation
```
</details>

<details>
<summary><strong>60. API Gateway Design</strong></summary>

```

Functions: auth, rate limiting, SSL termination, routing, caching, logging

Kong / AWS API Gateway / Nginx:
- Rate limit per user, per IP, per API key
- Request transformation: header injection, body modification
- Circuit breaker: stop forwarding to unhealthy backends
- Blue/green routing: split traffic by header or percentage
```
</details>

<details>
<summary><strong>61. Data Contracts</strong></summary>

```

Explicit agreements between data producers and consumers:
- Schema (field names, types, nullability)
- SLA (freshness, availability)
- Semantics (what does user_id mean?)
- Versioning (breaking vs non-breaking changes)

Tools: Soda, Great Expectations, dbt tests
Prevents silent breaking changes that only surface downstream hours later
```
</details>

<details>
<summary><strong>62. Hyperparameter Tuning at Scale</strong></summary>

```python

from ray import tune
from ray.tune.search.optuna import OptunaSearch

def train_fn(config):
    model = build_model(config["lr"], config["layers"])
    val_acc = train_and_eval(model)
    tune.report(val_accuracy=val_acc)

tuner = tune.Tuner(train_fn,
    param_space={
        "lr": tune.loguniform(1e-4, 1e-1),
        "layers": tune.randint(2, 8)
    },
    tune_config=tune.TuneConfig(search_alg=OptunaSearch(), num_samples=100))
results = tuner.fit()
```
</details>

<details>
<summary><strong>63. Write-Ahead Log (WAL)</strong></summary>

```

WAL: DB writes to append-only log first (fast sequential I/O), then data files
- Enables crash recovery: replay log after crash
- Enables replication: ship WAL to replicas
- Point-in-time recovery: restore to any past moment

Postgres WAL -> logical replication -> read replicas
Debezium: reads Postgres WAL -> CDC events -> Kafka
```
</details>

<details>
<summary><strong>64. Gossip Protocol and Leader Election</strong></summary>

```

Gossip: nodes randomly share state with peers; converges in O(log N) rounds
Used in: Cassandra, DynamoDB, memberlist

Leader Election (Raft):
1. Candidates request votes from majority
2. Elected leader sends heartbeats
3. Followers that miss heartbeats become candidates
4. Fencing tokens: prevent split-brain from old leaders
```
</details>

<details>
<summary><strong>65. Stateful Stream Processing</strong></summary>

```python

# Flink state types
# ValueState: single value per key
# ListState: list of values per key
# MapState: key-value map per key

# Watermarks: track event-time progress; allow late arrivals
# Checkpoint: save state to S3 periodically for fault tolerance
# Savepoint: manually triggered snapshot for migration/upgrades
```
</details>

<details>
<summary><strong>66. Real-Time Analytics with ClickHouse</strong></summary>

```sql

-- Columnar storage: fast aggregations on large datasets
SELECT
    toStartOfMinute(event_time) AS minute,
    uniq(user_id)               AS dau
FROM events
WHERE event_time > now() - INTERVAL 24 HOUR
GROUP BY minute
ORDER BY minute;

-- Ingestion: Kafka -> ClickHouse Kafka engine table
-- Materialized views: pre-aggregate on insert
```
</details>

<details>
<summary><strong>67. Graph Neural Networks for Fraud</strong></summary>

```

Node features: account age, transaction velocity, device count
Edge features: transaction amount, time delta, merchant category

GNN aggregates neighbor information to detect:
- Money mule networks (hub nodes)
- Card testing rings (burst of small transactions)
- Synthetic identity fraud (shared device/email/address)

Tools: PyTorch Geometric, DGL, GraphSAGE
```
</details>

<details>
<summary><strong>68. Model Cards and Responsible AI</strong></summary>

```

Model card documents:
- Intended use and out-of-scope uses
- Training data and known biases
- Performance metrics by demographic group
- Fairness metrics (equal opportunity, demographic parity)
- Limitations and recommendations

Fairness monitoring:
- Monitor TPR, FPR, precision per group in production
- Alert when group disparity exceeds threshold
```
</details>

<details>
<summary><strong>69. Orchestrating ML Workflows</strong></summary>

```python

# Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG('ml_pipeline', schedule_interval='@daily') as dag:
    ingest   = PythonOperator(task_id='ingest',    python_callable=ingest_data)
    features = PythonOperator(task_id='features',  python_callable=compute_features)
    train    = PythonOperator(task_id='train',      python_callable=train_model)
    evaluate = PythonOperator(task_id='evaluate',   python_callable=evaluate_model)
    deploy   = PythonOperator(task_id='deploy',     python_callable=deploy_if_better)
    ingest >> features >> train >> evaluate >> deploy

# Alternatives: Prefect, Dagster, Metaflow, Kubeflow Pipelines, ZenML
```
</details>

<details>
<summary><strong>70. Multi-Armed Bandit for Model Selection</strong></summary>

```python

import numpy as np

class ThompsonSampling:
    def __init__(self, n_models):
        self.alpha = np.ones(n_models)  # successes + 1
        self.beta  = np.ones(n_models)  # failures + 1

    def select_model(self):
        return np.argmax(np.random.beta(self.alpha, self.beta))

    def update(self, model_idx, reward):
        if reward: self.alpha[model_idx] += 1
        else:      self.beta[model_idx]  += 1
# Use to explore new models while exploiting known good ones
```
</details>

<details>
<summary><strong>71. Zero-Copy and High-Performance Data Transfer</strong></summary>

```

sendfile() syscall: file-to-socket without user-space copy
RDMA: GPU-to-GPU without CPU involvement
Apache Arrow: columnar in-memory format shared across languages
mmap: memory-mapped files (read without copying to process memory)
```
</details>

<details>
<summary><strong>72. Counting Bits and Probabilistic Data Structures</strong></summary>

```python

# HyperLogLog: count distinct elements with ~1% error, O(log log N) space
# Bloom Filter: probabilistic set membership (no false negatives, small false positives)
# Count-Min Sketch: approximate frequency counting

from pybloom_live import BloomFilter
bloom = BloomFilter(capacity=1000000, error_rate=0.001)
bloom.add("user_123")
"user_123" in bloom  # True (definite)
"user_999" in bloom  # False (definite negative)
```
</details>

<details>
<summary><strong>73. Idempotency in Distributed Systems</strong></summary>

```

Idempotency key: client sends unique key per request
Server: check key in DB before processing
  - If seen: return cached result
  - If new: process and store result with key

Critical for: payment APIs, order creation, email sending
Implementation: Redis SET NX with TTL, or DB UNIQUE constraint on key
```
</details>

<details>
<summary><strong>74. Replication Strategies</strong></summary>

```

Synchronous: primary waits for replica ACK
  + Strong consistency
  - Higher write latency

Asynchronous: primary returns immediately
  + Fast writes
  - Risk data loss on primary failure

Semi-synchronous: wait for at least one replica
Quorum: wait for majority (N/2 + 1) - used in Raft/Paxos
```

**75-100. Additional Topics Summary**
</details>

<details>
<summary><strong>75. Blue-Green Deployment</strong></summary>

Two identical envs; swap DNS/LB pointer. Zero downtime, instant rollback.
</details>

<details>
<summary><strong>76. Canary Release</strong></summary>

Route 1% -> 5% -> 20% -> 100% gradually. Monitor error rate at each step. Auto-rollback on regression.
</details>

<details>
<summary><strong>77. Circuit Breaker States</strong></summary>

- Closed: normal operation

- Open: fail-fast after N failures (no calls to downstream)
- Half-open: allow one test request after timeout; close if succeeds
</details>

<details>
<summary><strong>78. Bulkhead Pattern</strong></summary>

Separate thread pools per downstream service. One slow dependency can't exhaust resources for others.
</details>

<details>
<summary><strong>79. MVCC (Multi-Version Concurrency Control)</strong></summary>

Readers never block writers. Each transaction sees consistent snapshot. Used by Postgres, MySQL InnoDB.
</details>

<details>
<summary><strong>80. Apache Kafka Exactly-Once Semantics</strong></summary>

- Producer: `enable.idempotence=true` + transactions

- Consumer: commit offset AFTER successful processing
- End-to-end: transactional read-process-write atomically
</details>

<details>
<summary><strong>81. Data Mesh Architecture</strong></summary>

- Domain-oriented ownership (team owns their data)

- Data as a product (quality, discoverability SLAs)
- Self-serve platform (shared infra)
- Federated governance (global standards, local autonomy)
</details>

<details>
<summary><strong>82. Content-Based vs Collaborative Filtering</strong></summary>

- Content-based: recommend similar items to what user liked (cold start for users, not items)

- Collaborative: recommend what similar users liked (cold start for new items)
- Hybrid: combine both (most production systems)
</details>

<details>
<summary><strong>83. GPU Memory Management for LLMs</strong></summary>

- KV Cache: grows with sequence length and batch size

- PagedAttention (vLLM): manage KV cache like OS pages
- Flash Decoding: split KV across SMs for faster attention
- Quantization (AWQ/GPTQ): 4-bit weights, 4x memory reduction
</details>

<details>
<summary><strong>84. Distributed Tracing Propagation</strong></summary>

- Trace ID: unique per request (propagated across services)

- Span ID: unique per operation within trace
- Baggage: key-value pairs propagated with trace
- W3C TraceContext: standard HTTP header format
- Tools: Jaeger, Zipkin, Tempo (Grafana)
</details>

<details>
<summary><strong>85. Event-Driven Architecture Patterns</strong></summary>

- Event notification: tell subscribers something happened

- Event-carried state transfer: include full state in event
- Event sourcing: events ARE the system of record
- Domain events vs integration events distinction
</details>

<details>
<summary><strong>86. Kubernetes Operators for ML</strong></summary>

```yaml

# Custom Resource for distributed training
apiVersion: kubeflow.org/v1
kind: PyTorchJob
spec:
  pytorchReplicaSpecs:
    Master: {replicas: 1, template: {spec: {containers: [{image: trainer:v1}]}}}
    Worker: {replicas: 7, template: {spec: {containers: [{image: trainer:v1}]}}}
```
</details>

<details>
<summary><strong>87. Zero-Downtime Database Migrations</strong></summary>

1. Expand: add new column (nullable, no default required)

2. Migrate: backfill data in batches (avoid table lock)
3. Switch: application uses new column
4. Contract: drop old column
Never: ALTER TABLE with NOT NULL on large tables (full table lock)
</details>

<details>
<summary><strong>88. Read-Your-Writes Consistency</strong></summary>

After a write, the same user's subsequent reads see that write.

Solutions:
- Route user's reads to primary for 1 min after writes
- Sticky routing to same replica
- Version token: read-your-writes token guarantees replica is caught up
</details>

<details>
<summary><strong>89. Multi-Level Caching for ML Features</strong></summary>

```

L1: In-process dict (last 100 users, microseconds, lost on restart)
L2: Local Redis (last 10K users, milliseconds)
L3: Redis cluster (all users, milliseconds, shared across instances)
L4: S3 pre-computed batch (historical features, seconds)
L5: Feature pipeline DB (ground truth, seconds)
```
</details>

<details>
<summary><strong>90. Handling Thundering Herd</strong></summary>

```python

import threading, time, random

_cache = {}
_locks = {}

def get_with_lock(key, compute_fn, ttl=60):
    if key in _cache and _cache[key][1] > time.time():
        return _cache[key][0]
    if key not in _locks:
        _locks[key] = threading.Lock()
    with _locks[key]:
        if key not in _cache or _cache[key][1] <= time.time():
            value = compute_fn()
            _cache[key] = (value, time.time() + ttl + random.uniform(0, ttl*0.1))
    return _cache[key][0]
```
</details>

<details>
<summary><strong>91. Shard Management with Vitess</strong></summary>

```

Vitess: MySQL sharding proxy (used by YouTube, Slack, GitHub)
- VSchema: virtual schema that spans shards
- VTGate: query router (handles cross-shard queries)
- VTTablet: wraps each MySQL instance
- Resharding: online shard splitting without downtime
```
</details>

<details>
<summary><strong>92. Model Registry Workflow</strong></summary>

```

Experiment run -> Staging (validation) -> Production -> Archived

Champion/Challenger:
- Production serves champion model
- Shadow-evaluates challenger in parallel
- Promote challenger if metrics better after N samples

MLflow stages: None -> Staging -> Production -> Archived
```
</details>

<details>
<summary><strong>93. Embedding Cache for LLM Applications</strong></summary>

```python

import redis, json, hashlib
import numpy as np

r = redis.Redis()

def get_embedding(text, embed_fn, ttl=86400):
    key = "emb:" + hashlib.sha256(text.encode()).hexdigest()
    cached = r.get(key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32)
    embedding = embed_fn(text)
    r.setex(key, ttl, embedding.tobytes())
    return embedding
# Reduces LLM API calls by 80-90% for repeated queries
```
</details>

<details>
<summary><strong>94. Designing a URL Shortener</strong></summary>

```

Functional: shorten URL, redirect, analytics
Non-functional: 100M URLs/day writes, 10B reads/day, 5yr retention

Encoding: base62 (a-z A-Z 0-9), 7 chars = 62^7 = 3.5T unique URLs
Storage: 7 + ~200 bytes per URL * 100M/day * 365 * 5yr = ~130TB

Architecture:
POST /shorten -> hash or counter -> store in DB -> return short_url
GET /{code}   -> DB lookup (cache in Redis) -> 301 redirect

Analytics: Kafka stream of redirects -> Flink aggregation -> ClickHouse
```
</details>

<details>
<summary><strong>95. Designing Instagram Feed</strong></summary>

```

Pull model: read from followees at load time (fresh, slow for many followees)
Push model: fan-out on write (fast reads, expensive for celebrities)
Hybrid: push for regular users (<10K followers), pull for celebrities

Feed storage: Redis sorted set (ZADD feed:{user_id} {timestamp} {post_id})
Pagination: cursor-based (return last seen timestamp)
Media storage: S3 + CloudFront CDN
```
</details>

<details>
<summary><strong>96. Webhook Reliability</strong></summary>

```python

# Guaranteed delivery with retries
def send_webhook(url, payload, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=5)
            if r.status_code == 200:
                return True
            elif r.status_code < 500:  # client error, don't retry
                return False
        except Exception:
            pass
        time.sleep(2 ** attempt)  # exponential backoff
    # Move to dead-letter queue for manual inspection
    dlq.publish(payload)
    return False
```
</details>

<details>
<summary><strong>97. Cross-Service Authentication</strong></summary>

```

Service-to-service: mTLS (mutual TLS) certificates or JWT service tokens
JWT verification: validate signature, expiry, issuer, audience
API key rotation: zero-downtime rotation via grace period (accept both old+new)
Secret management: HashiCorp Vault, AWS Secrets Manager (never in env vars or code)
```
</details>

<details>
<summary><strong>98. Serverless for ML Inference</strong></summary>

```

AWS Lambda / Google Cloud Functions:
+ Zero infrastructure management, auto-scaling, pay-per-request
- Cold starts (100ms-5s), memory limits (10GB max), no GPU

Use cases: low-traffic models, event-triggered inference, preprocessing
Not suitable: GPU models, sustained high-throughput, streaming

Lambda container images: up to 10GB, pre-load model in global scope
Provisioned concurrency: eliminate cold starts (higher cost)
```
</details>

<details>
<summary><strong>99. Designing a Notification System</strong></summary>

```

Channels: push (mobile), email, SMS, in-app

Architecture:
[Event] -> [Notification Service] -> [Channel Router] -> [Channel Handlers]
                    |
             [User Preferences DB]  (opt-outs, quiet hours, frequency caps)

Deduplication: cache notification_id for 24h to prevent duplicates
Rate limiting: max N notifications per user per hour per channel
Prioritization: critical (OTP) > transactional > marketing
```
</details>

<details>
<summary><strong>100. Designing a Distributed ML Training Platform</strong></summary>

```

Components:
1. Job Scheduler: queue jobs, assign GPU resources (FIFO, fair-share, priority)
2. Data Loading: S3 -> FUSE mount or streaming (petastorm, webdataset)
3. Distributed Training: PyTorch DDP / DeepSpeed ZeRO / FSDP
4. Checkpointing: save to S3 every N steps (resume on preemption)
5. Experiment Tracking: MLflow / W&B (metrics, params, artifacts)
6. Model Registry: versioned model store with promotion workflow
7. Hyperparameter Optimization: Ray Tune / Optuna (parallel trials)
8. Cost Management: spot instances with automatic checkpoint/resume

Job spec:
{
  "image": "trainer:v2",
  "gpus": 8,
  "nodes": 4,
  "command": "torchrun --nproc_per_node=8 train.py",
  "data_path": "s3://data/train",
  "checkpoint_path": "s3://checkpoints/job-123"
}
```
</details>
---

## System Design Cheat Sheet

| Pattern | Problem Solved | Key Trade-off |
|---------|---------------|---------------|
| Cache-aside | Slow DB reads | Stale data risk |
| Write-through | Cache consistency | Higher write latency |
| Circuit Breaker | Cascade failures | False positives |
| Saga | Distributed transactions | Eventual consistency |
| CQRS | Read/write scalability | Sync complexity |
| Event Sourcing | Audit trail | Query complexity |
| Consistent Hashing | Node add/remove | Key distribution skew |
| Bulkhead | Resource isolation | Resource underutilization |
| Strangler Fig | Monolith migration | Parallel operation cost |
| Two-Tower | Retrieval at scale | Approximate matching |

## Summary in simple points

- Start with workload, correctness, latency, availability, cost and authority requirements; estimate scale before drawing infrastructure.
- Separate scaling, availability and reliability. SLOs and error budgets need explicit denominators and measurement windows.
- Load balancing, discovery, gateways, service meshes and REST/GraphQL/gRPC have different routing and interface responsibilities.
- Caching needs identity, freshness and invalidation across layers. Handle stampedes, hot keys and embedding/feature cache versioning.
- Partitioning, sharding, consistent hashing and replication trade placement, rebalancing and failure behaviour; account for hot partitions.
- Database indexes, connection pools, MVCC, WAL, OLTP/OLAP and storage layouts support different workloads and consistency needs.
- Queues, streams, backpressure, schema evolution and data contracts need durable identity, ordering and recovery semantics.
- Distributed locks, leader election, gossip, sagas, CQRS and event sourcing address distinct coordination/state problems.
- Idempotency is an application contract; exactly-once claims apply to a stated boundary and do not automatically include arbitrary external effects.
- Feature stores, real-time pipelines, lakes/warehouses and lineage must preserve historical availability and consistent feature definitions.
- Recommendation, search, fraud, feed, notification and URL-shortening designs require task-specific access patterns and failure handling.
- ML platforms need experiment tracking, tuning, workflow orchestration, model registries, CI/CD and explicit promotion gates.
- Training/serving scale depends on hardware, batching, memory and communication. LLM serving adds tokenizer, KV-cache and generation constraints.
- Model monitoring, drift, class imbalance, delayed labels, online learning and bandits require suitable feedback and statistical design.
- Shadow, canary and blue-green deployments have different evidence and rollback requirements. Database migrations must preserve compatibility.
- Circuit breakers, bulkheads, graceful failure, chaos tests and disaster recovery need concrete triggers and tested recovery paths.
- Metrics, logs and distributed traces complement one another; preserve correlation while bounding sensitive data and cardinality.
- Multi-tenancy, authentication, privacy, regional deployment and model documentation require controls across the full data lifecycle.
- Kubernetes resources/operators, serverless, edge/CDN, managed databases and streaming vendors are implementation choices to evaluate against constraints.
- Probabilistic data structures, zero-copy transfer and specialised analytics can help specific bottlenecks; validate approximation and memory assumptions.
