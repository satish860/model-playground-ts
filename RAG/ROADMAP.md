# RAG Module - Agentic Search Roadmap

## 📖 Overview

This module implements **agentic documentation search** - an alternative to traditional vector-based RAG that uses tool-calling agents to iteratively search and synthesize information from documentation.

### The Problem

**Use Case**: Documentation Q&A system for Anthropic's Claude documentation

**Users**:
- Developers learning Claude API
- Support engineers answering customer questions
- Technical writers verifying documentation

**Challenge**:
- 75% of questions require information from 2+ documentation chunks
- Users need accurate answers with source citations
- Documentation is constantly updated

### Why Agentic Search?

**Traditional RAG Limitations:**
- Requires pre-computed embeddings (indexing overhead)
- Fixed retrieval strategy (no adaptation)
- Black box (can't see what was retrieved)
- Sync issues (index may be stale)

**Agentic Search Advantages:**
- No embeddings needed - just keyword search
- Agent decides search strategy dynamically
- Transparent (see all tool calls)
- Always searches latest data
- Better for complex multi-part questions

**Trade-offs:**
- Higher latency (multiple LLM calls)
- Higher token cost (agent reasoning)
- Requires capable reasoning model

---

## 📊 The Data

### Documentation Corpus
**File**: `data/anthropic_docs.json` (232 chunks)

**Structure**:
```json
{
  "chunk_link": "https://docs.claude.com/en/docs/welcome#get-started",
  "chunk_heading": "Get started",
  "text": "Full documentation text with heading..."
}
```

**Coverage**: Anthropic's official Claude documentation including:
- Getting Started (intro, quickstart)
- API Reference (endpoints, parameters)
- Models (comparison, capabilities)
- Use Cases (classification, summarization, etc.)
- Prompt Engineering (techniques, best practices)
- Tools & Evaluation (testing, monitoring)

### Summary-Indexed Docs
**File**: `data/anthropic_summary_indexed_docs.json` (232 chunks)

Same chunks with added `summary` field (2-3 sentence AI-generated summaries). Can be used for faster initial filtering.

### Evaluation Dataset
**File**: `data/evaluation/docs_evaluation_dataset.json` (100 Q&A pairs)

**Structure**:
```json
{
  "id": "efc09699",
  "question": "How can you create multiple test cases for an evaluation?",
  "correct_chunks": [
    "https://docs.claude.com/en/docs/test-and-evaluate/eval-tool#creating-test-cases",
    "https://docs.claude.com/en/docs/build-with-claude/develop-tests#building-evals"
  ],
  "correct_answer": "To create multiple test cases in the Anthropic Evaluation tool..."
}
```

**Statistics**:
- 100 questions covering all documentation topics
- Average question: 20 words
- Average answer: 36 words
- 75% require 2 chunks, 17% require 1 chunk, 8% require 3+ chunks
- Topics: API (28), Prompt Engineering (19), Evaluation (13), Models (11), Embeddings (4)

---

## 🎯 Three Progressive Levels

### Level 1: Simple Agent (Baseline)
**Complexity**: Low
**Implementation Time**: 1-2 days
**Expected Accuracy**: 60-70%

**Capabilities**:
- Basic keyword search
- Read specific chunks
- Simple answer synthesis
- 1-3 tool calls per question

**Agent Loop**:
```
User Question
    ↓
Agent: "I need to search for X"
    ↓
Tool: chunk_search("X") → Results
    ↓
Agent: "Let me read the top result"
    ↓
Tool: chunk_read([42]) → Full chunk
    ↓
Agent: "Now I can answer"
    ↓
Final Answer with citation
```

**Limitations**:
- No multi-step reasoning
- No search refinement
- May miss relevant chunks
- No self-correction

---

### Level 2: Planning Agent (Enhanced)
**Complexity**: Medium
**Implementation Time**: 3-5 days
**Expected Accuracy**: 75-85%

**Capabilities**:
- Explicit planning phase before searching
- Multi-step search refinement
- Context expansion (get surrounding chunks)
- Shows reasoning with `<think>` tags
- 3-7 tool calls per question

**Agent Loop**:
```
User Question
    ↓
Agent: <think>This question asks about X and Y.
       I should search for both separately then combine.</think>
    ↓
Tool: chunk_search("X") → [chunk_10, chunk_15]
    ↓
Tool: chunk_search("Y") → [chunk_42, chunk_55]
    ↓
Agent: <think>I found info about X but Y results look incomplete.
       Let me try a related term.</think>
    ↓
Tool: chunk_search("Y alternative term") → [chunk_67]
    ↓
Tool: chunk_read([10, 42, 67]) → Read all relevant chunks
    ↓
Agent: "Based on chunks 10, 42, and 67..."
    ↓
Final Answer with multiple citations
```

**Enhancements**:
- Pattern matching with regex
- Filter by heading/link patterns
- Progressive search narrowing
- Explicit reasoning traces

---

### Level 3: Advanced Agent (Production-Ready)
**Complexity**: High
**Implementation Time**: 5-7 days
**Expected Accuracy**: 85-95%

**Capabilities**:
- Self-correction when searches fail
- Synonym expansion and related term search
- Search strategy library
- Performance optimization (caching, parallel calls)
- Comprehensive evaluation metrics

**Agent Loop**:
```
User Question
    ↓
Agent: <think>Multi-part question. Plan:
       1. Search for main concept
       2. Search for specific details
       3. Cross-reference if needed</think>
    ↓
Tool: chunk_search("main concept") → [chunk_88]
    ↓
Tool: chunk_read([88])
    ↓
Agent: <think>This mentions the concept but lacks details.
       The details are probably in a different section.
       Let me try synonyms.</think>
    ↓
Tool: chunk_search("synonym 1") → No results
Tool: chunk_search("synonym 2") → [chunk_102, chunk_103]
    ↓
Tool: chunk_read([102, 103])
    ↓
Agent: <think>Found it! The details were in the pricing section,
       not where I initially expected.</think>
    ↓
Final Answer: "According to the pricing documentation (chunk 102)
               and configuration guide (chunk 103)..."
```

**Advanced Features**:
- Search strategy patterns (progressive narrowing, section-aware search)
- Self-assessment of answer confidence
- Recognition when information is not in docs
- Performance metrics tracking
- Hybrid mode (optional lightweight embeddings for initial filtering)

---

## 🛠️ Implementation Architecture

### Tools (3 core + 2 optional)

#### Core Tools

**1. chunk_search**
```python
def chunk_search(
    query: str,
    case_sensitive: bool = False,
    search_in: str = "all",  # all, heading, text, summary
    max_results: int = 10
) -> List[Dict]:
    """
    Search documentation chunks by keyword.

    Returns:
    [
        {
            "chunk_id": 42,
            "chunk_link": "https://...",
            "chunk_heading": "Streaming",
            "snippet": "...streaming responses allow...",
            "match_count": 5,
            "match_score": 0.5
        }
    ]
    """
```

**2. chunk_read**
```python
def chunk_read(chunk_ids: List[int]) -> List[Dict]:
    """
    Read full content of specific chunks.

    Returns:
    [
        {
            "chunk_id": 42,
            "chunk_link": "https://...",
            "chunk_heading": "Streaming",
            "text": "Full chunk text..."
        }
    ]
    """
```

**3. chunk_filter**
```python
def chunk_filter(
    heading_pattern: Optional[str] = None,
    link_pattern: Optional[str] = None
) -> List[int]:
    """
    Filter chunks by heading or link pattern.

    Returns: List of matching chunk IDs
    """
```

#### Optional Tools (Level 3)

**4. chunk_context**
```python
def chunk_context(
    chunk_id: int,
    before: int = 2,
    after: int = 2
) -> List[Dict]:
    """Get surrounding chunks for context."""
```

**5. chunk_stats**
```python
def chunk_stats(group_by: str = "link_section") -> Dict:
    """Get statistics about documentation structure."""
```

---

### Agent Loop Structure

**Basic Pattern**:
```python
def agent_loop(question: str, max_iterations: int = 5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    for iteration in range(max_iterations):
        # 1. Agent decides next action
        response = llm.chat(messages, tools=available_tools)

        # 2. Execute tool calls
        if response.tool_calls:
            tool_results = execute_tools(response.tool_calls)
            messages.append(response.message)
            messages.append(tool_results)
            continue  # Loop again

        # 3. No more tools needed - return answer
        return response.content

    return "Max iterations reached"
```

**System Prompt (Level 1)**:
```
You are a helpful documentation search assistant.

Your task: Answer questions about Claude documentation using the available tools.

Tools:
- chunk_search: Find chunks containing keywords
- chunk_read: Read full content of specific chunks
- chunk_filter: Filter chunks by heading/link patterns

Strategy:
1. Search for relevant keywords
2. Read the most promising chunks
3. Provide a clear answer with source citations

Always cite your sources using chunk links.
```

**System Prompt (Level 2 - with planning)**:
```
You are an expert documentation search assistant.

Process:
1. PLAN: Think about what information you need
2. SEARCH: Use tools to find relevant chunks (start broad, refine)
3. READ: Examine promising chunks
4. REFLECT: Do you have enough info? If not, search again
5. ANSWER: Provide complete answer with citations

Use <think>...</think> tags to show your reasoning process.
```

---

## 📈 Evaluation Methodology

### Metrics

**1. Accuracy**
```python
correct = 0
for qa_pair in test_set:
    answer = agent(qa_pair['question'])
    if answer_matches(answer, qa_pair['correct_answer']):
        correct += 1
accuracy = correct / len(test_set)
```

**2. Chunk Recall**
```python
# Did agent retrieve the correct chunks?
correct_chunks_found = 0
for qa_pair in test_set:
    retrieved_chunks = agent.get_retrieved_chunks()
    if all(c in retrieved_chunks for c in qa_pair['correct_chunks']):
        correct_chunks_found += 1
chunk_recall = correct_chunks_found / len(test_set)
```

**3. Efficiency**
```python
avg_tool_calls = sum(agent.tool_call_count for each question) / num_questions
avg_latency = sum(agent.latency for each question) / num_questions
avg_tokens = sum(agent.token_usage for each question) / num_questions
```

**4. Answer Quality (LLM-as-Judge)**
```python
def evaluate_answer_quality(generated, reference):
    """Use LLM to judge answer quality on scale 1-5"""
    prompt = f"""
    Question: {question}
    Reference Answer: {reference}
    Generated Answer: {generated}

    Rate the generated answer (1-5):
    - Accuracy: Factually correct?
    - Completeness: Covers all points?
    - Clarity: Easy to understand?

    Overall Score (1-5):
    """
    return llm.evaluate(prompt)
```

### Comparison Baselines

**Baseline 1**: No tools (direct answer from model)
**Baseline 2**: Single-shot RAG with embeddings
**Baseline 3**: Agentic search (this implementation)

**Expected Results**:
```
Approach              Accuracy  Chunk Recall  Avg Latency  Cost
--------------------------------------------------------------------
No Tools (baseline)   40-50%    N/A           1s           $
Vector RAG            70-80%    75-85%        2-3s         $$
Level 1 Agent         60-70%    60-70%        5-10s        $$$
Level 2 Agent         75-85%    80-90%        10-20s       $$$$
Level 3 Agent         85-95%    90-95%        15-30s       $$$$$
```

---

## 🗺️ Implementation Roadmap

### Phase 1: Setup & Core Tools (Week 1)

**Day 1-2: Tool Implementation**
- [ ] Create `tools/` folder structure
- [ ] Implement `chunk_search.py` with keyword matching
- [ ] Implement `chunk_read.py` for reading chunks
- [ ] Implement `chunk_filter.py` for pattern filtering
- [ ] Write unit tests for each tool

**Day 3-4: Simple Agent**
- [ ] Create `simple_agent.py` with basic loop
- [ ] Integrate with OpenRouter (MiniMax M2)
- [ ] Test on 10 sample questions manually
- [ ] Debug and refine tool schemas

**Day 5: Testing & Evaluation**
- [ ] Create `test_agent.py` for automated testing
- [ ] Run on 20 questions from test set
- [ ] Measure baseline accuracy
- [ ] Document limitations and failures

**Deliverables**:
- ✅ Working Level 1 agent
- ✅ 3 core tools functional
- ✅ Baseline accuracy measured (~60-70%)

---

### Phase 2: Planning Agent (Week 2)

**Day 1-2: Enhanced Tools**
- [ ] Add regex support to `chunk_search`
- [ ] Implement `chunk_context` tool
- [ ] Add parallel tool execution
- [ ] Optimize for speed

**Day 3-4: Planning Agent**
- [ ] Create `planning_agent.py` with enhanced prompts
- [ ] Add planning phase before search
- [ ] Implement multi-step refinement
- [ ] Show reasoning with `<think>` tags

**Day 5: Evaluation**
- [ ] Create `evaluator.py` for comprehensive metrics
- [ ] Run on full 100-question test set
- [ ] Compare to Level 1 baseline
- [ ] Analyze failure cases

**Deliverables**:
- ✅ Level 2 planning agent
- ✅ Enhanced tool suite
- ✅ Comprehensive evaluation (~75-85% accuracy)

---

### Phase 3: Advanced Features (Week 3)

**Day 1-2: Self-Correction**
- [ ] Implement search strategy library
- [ ] Add synonym expansion
- [ ] Add section-aware search
- [ ] Implement confidence scoring

**Day 3-4: Optimization**
- [ ] Add result caching
- [ ] Parallel tool execution
- [ ] Early stopping when confident
- [ ] Performance profiling

**Day 5: Production Polish**
- [ ] Comprehensive benchmarking
- [ ] Error handling and edge cases
- [ ] Documentation and examples
- [ ] Comparison report (vs RAG)

**Deliverables**:
- ✅ Level 3 advanced agent
- ✅ Production-ready system
- ✅ Full evaluation report (~85-95% accuracy)

---

### Phase 4: Hybrid Approach (Optional)

**Goal**: Best of both worlds - combine lightweight RAG with agentic refinement

**Implementation**:
```python
def hybrid_search(question: str):
    # Stage 1: Fast initial filtering with embeddings
    candidate_chunks = embedding_search(question, top_k=20)

    # Stage 2: Agentic refinement with tools
    agent = SearchAgent(candidate_chunks)
    answer = agent.search_and_answer(question)

    return answer
```

**Benefits**:
- Fast initial filtering (embeddings)
- Smart refinement (agentic)
- Best accuracy with managed cost

---

## 📝 File Structure (Final)

```
RAG/
├── data/                                  # Data files (downloaded)
│   ├── anthropic_docs.json               # 232 chunks
│   ├── anthropic_summary_indexed_docs.json
│   └── evaluation/
│       └── docs_evaluation_dataset.json  # 100 Q&A pairs
│
├── tools/                                 # Tool implementations
│   ├── __init__.py
│   ├── chunk_search.py                   # Keyword search
│   ├── chunk_read.py                     # Read full chunks
│   ├── chunk_filter.py                   # Filter by pattern
│   ├── chunk_context.py                  # Get surrounding chunks (L3)
│   └── chunk_stats.py                    # Stats tool (L3)
│
├── simple_agent.py                        # Level 1: Basic agent
├── planning_agent.py                      # Level 2: Planning agent
├── advanced_agent.py                      # Level 3: Advanced agent
├── evaluator.py                           # Evaluation framework
├── test_agent.py                          # Test suite
├── benchmark.py                           # Performance benchmarking
├── search_strategies.py                   # Strategy library (L3)
│
├── requirements.txt                       # Dependencies
├── ROADMAP.md                            # This file
└── README.md                             # Usage documentation
```

---

## 🎯 Success Criteria

### Level 1 (Minimum Viable)
- ✅ Agent can call tools successfully
- ✅ Answers 60%+ of test questions correctly
- ✅ Shows transparent tool usage
- ✅ Completes in <10s per question
- ✅ No crashes or errors

### Level 2 (Production-Ready)
- ✅ Answers 75%+ of test questions correctly
- ✅ Multi-step reasoning demonstrated
- ✅ Search refinement working
- ✅ Shows reasoning process
- ✅ Handles complex multi-part questions

### Level 3 (Advanced)
- ✅ Answers 85%+ of test questions correctly
- ✅ Self-correction implemented
- ✅ Synonym expansion working
- ✅ Performance optimizations active
- ✅ Comprehensive evaluation metrics
- ✅ Competitive with or better than vector RAG

---

## 🔑 Key Insights

### When to Use Agentic Search
- ✅ Complex multi-step questions
- ✅ Frequently updated documentation
- ✅ Need transparent search process
- ✅ Have capable reasoning model
- ✅ Accuracy > Speed

### When to Use Traditional RAG
- ✅ Simple factual questions
- ✅ Large static corpus
- ✅ Speed > Accuracy
- ✅ Cost-sensitive
- ✅ Don't need transparency

### Hybrid Is Best
Combine both: Use embeddings for fast initial filtering, then agentic refinement for complex questions.

---

## 📚 References

- [Claude Cookbook - RAG Guide](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/retrieval_augmented_generation/guide.ipynb)
- [Anthropic Research - Agentic Search](https://www.anthropic.com/research)
- [Kimi CLI - File Tools](https://github.com/MoonshotAI/kimi-cli/tree/main/src/kimi_cli/tools/file)

---

**Status**: Phase 1 - Ready to implement! 🚀
