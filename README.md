# AI Model Playground

Multi-language model testing framework for evaluating and comparing AI models through [OpenRouter](https://openrouter.ai/). Test any of 200+ models with both TypeScript and Python implementations.

## 🎯 What is This?

A playground for rapid experimentation with different AI models. Switch between models with a single configuration change and compare their capabilities across various tasks:

- 💬 **Text Generation** - Basic prompts and responses
- 🌊 **Streaming** - Real-time token-by-token output
- 📊 **Structured Output** - JSON generation with validation
- 💭 **Chat** - Multi-turn conversations with context
- 🤖 **Agents** - Tool calling and function execution
- 🎯 **Classification** - Real-world ML task with RAG and Chain-of-Thought
- 📄 **Summarization** - Long document summarization with progressive techniques

Currently configured for **MiniMax M2**, but designed to work with any OpenRouter-supported model (Claude, GPT-4, Gemini, Llama, etc.).

---

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ (for TypeScript examples)
- Python 3.8+ (for Python examples)
- OpenRouter API key ([Get one free](https://openrouter.ai/keys))

### Installation

1. **Clone and setup:**
```bash
git clone <your-repo-url>
cd model-playground-ts
```

2. **Configure API key:**
```bash
# Copy example and add your key
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=your_key_here
```

3. **Install dependencies:**

**TypeScript:**
```bash
npm install
```

**Python:**
```bash
cd python
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

**Classification module:**
```bash
cd classification
pip install -r requirements.txt
```

---

## 📖 Usage Examples

### TypeScript

```bash
npm run start          # Basic text generation
npm run stream         # Streaming responses
npm run structured     # JSON output
npm run chat           # Interactive chat
npm run agent          # Agent with tools
```

**Example Output:**
```
Prompt: What is OpenRouter?
Response: OpenRouter is a unified API gateway that provides access to 200+ AI models...
```

### Python

```bash
cd python
python examples/basic_text.py    # Basic generation
python examples/streaming.py     # Streaming
python examples/structured.py    # JSON output
python examples/chat_cli.py      # Interactive chat
python examples/agent.py         # Agent with tools
```

### Classification Module

Progressive complexity demonstrating real-world ML task:

```bash
cd classification

# Three approaches with increasing accuracy:
python simple_classify.py   # Baseline: ~70% accuracy
python rag_classify.py      # RAG: ~85% accuracy
python cot_classify.py      # RAG + CoT: ~95%+ accuracy
```

**What it does:** Classifies insurance support tickets into 10 categories using different techniques:
- **Simple**: Just prompts and category definitions
- **RAG**: Retrieves 5 similar examples using vector embeddings
- **CoT**: RAG + explicit step-by-step reasoning (shows model's thinking!)

### Summarization Module

Progressive techniques for condensing long documents:

```bash
cd summarization

# Three approaches (Coming Soon):
python simple_summarize.py    # Basic: Simple bullet-point summary
python guided_summarize.py     # Structured: Guided field extraction
python chunking_summarize.py   # Advanced: Meta-summarization with chunking
```

**What it does:** Summarizes lengthy legal documents (lease agreements) using different techniques:
- **Simple**: Basic "summarize this" prompt
- **Guided**: Structured extraction (parties, dates, obligations, clauses)
- **Chunking**: Break document into chunks, summarize each, then synthesize (best for 50+ page docs)

**Status:** ⏳ Roadmap created, implementation coming soon. See `summarization/ROADMAP.md` for detailed plan.

---

## 🧩 Project Structure

```
model-playground-ts/
├── src/                    # TypeScript examples
├── python/examples/        # Python examples
├── classification/         # Real-world classification module
│   ├── simple_classify.py  # Baseline approach
│   ├── rag_classify.py     # Vector RAG
│   ├── cot_classify.py     # Chain-of-Thought
│   └── data/               # Training/test data
├── summarization/          # Document summarization module
│   ├── ROADMAP.md          # Detailed implementation plan
│   └── data/               # Lease documents (to be downloaded)
├── .env                    # API keys (create this)
└── package.json            # Dependencies
```

---

## 🔄 Testing Different Models

### Switch Models Easily

**TypeScript** - Change model in any file:
```typescript
model: openrouter('anthropic/claude-3.5-sonnet')
// or
model: openrouter('openai/gpt-4')
// or
model: openrouter('google/gemini-pro')
```

**Python** - Update `python/utils/config.py`:
```python
MODEL_FREE = "anthropic/claude-3.5-sonnet"
```

**Classification** - Update each classifier:
```python
MODEL = "anthropic/claude-3.5-sonnet"  # Test classification accuracy across models
```

### Available Models

OpenRouter provides access to 200+ models including:
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5
- **Google**: Gemini Pro, Gemini Ultra
- **Meta**: Llama 3.1, Llama 3.2
- **Mistral**: Mistral Large, Mixtral
- **MiniMax**: MiniMax M2 (current default)
- And many more...

See [OpenRouter models](https://openrouter.ai/models) for the full list.

---

## 🌟 Features

### Basic Capabilities
- ✅ Text generation
- ✅ Streaming responses
- ✅ Structured JSON output (with Zod/Pydantic validation)
- ✅ Multi-turn chat with history
- ✅ Interactive CLI interfaces

### Advanced Features
- ✅ **Tool Calling** - Agents that can execute functions (calculator, weather, time, etc.)
- ✅ **Thinking Process** - Some models (like MiniMax M2) expose internal reasoning
- ✅ **RAG** - Retrieval-Augmented Generation with vector embeddings
- ✅ **Chain-of-Thought** - Explicit step-by-step reasoning for better accuracy

### Real-World Use Case: Classification

The `classification/` module demonstrates a practical ML task:

**Problem:** Automatically categorize insurance support tickets into 10 categories

**Three Approaches:**
1. **Simple Prompts** → 70% accuracy
2. **RAG (Vector Similarity)** → 85% accuracy
3. **RAG + Chain-of-Thought** → 95%+ accuracy

**Why this matters:** Shows how prompt engineering, RAG, and reasoning techniques significantly improve real-world performance.

---

## 🎓 Learning Path

**Start here:**
1. Run `npm run start` (TypeScript) or `python examples/basic_text.py` (Python)
2. Try `npm run stream` to see streaming in action
3. Run `npm run chat` for interactive conversation

**Next level:**
4. Try `npm run structured` for JSON output
5. Run `npm run agent` to see tool calling (MiniMax M2 shows thinking process!)

**Advanced:**
6. Explore the classification module:
   - Run `simple_classify.py` for baseline
   - Run `rag_classify.py` to see RAG in action
   - Run `cot_classify.py` to see the model's reasoning!

**Experiment:**
7. Switch to a different model (Claude, GPT-4, etc.) and compare results
8. Try the same classification task with different models
9. Build your own use case!

---

## 🔧 Configuration

### Environment Variables

Required in `.env`:
```env
OPENROUTER_API_KEY=your_key_here
```

### Model Selection

- **Free tier models**: Append `:free` to model name (e.g., `minimax/minimax-m2:free`)
- **Paid models**: Use regular model name (e.g., `anthropic/claude-3.5-sonnet`)
- **Rate limits**: Free tiers have daily limits; paid tiers have higher limits

### OpenRouter Features

- **Credits**: Add credits at [OpenRouter billing](https://openrouter.ai/settings/billing)
- **Usage tracking**: Monitor at [OpenRouter activity](https://openrouter.ai/activity)
- **Model comparison**: [OpenRouter models page](https://openrouter.ai/models)

---

## 📊 Classification Module Details

### Data

- **Source**: [Claude Cookbook - Classification Guide](https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/classification)
- **Training**: 68 labeled insurance support tickets
- **Testing**: 68 unlabeled tickets for evaluation
- **Categories**: 10 insurance categories (Billing, Claims, Policy, etc.)

### Approaches

#### 1. Simple Prompt (`simple_classify.py`)
```
Categories → Ticket → Classification
```
- No examples, just category definitions
- Fast but less accurate (~70%)

#### 2. RAG (`rag_classify.py`)
```
Categories → [Find 5 similar examples] → Ticket → Classification
```
- Uses sentence-transformers for embeddings
- In-memory vector store with cosine similarity
- Much better accuracy (~85%)

#### 3. Chain-of-Thought (`cot_classify.py`)
```
Categories → [Find 5 similar examples] → Ticket → Think Step-by-Step → Classification
```
- RAG + explicit reasoning prompt
- Shows model's thinking process for each classification
- Highest accuracy (~95%+)
- Demonstrates how reasoning improves performance

---

## 🤝 Contributing

This is a testing playground. Feel free to:
- Add new examples (image generation, embeddings, etc.)
- Test new models and compare results
- Build new use cases (summarization, Q&A, code generation, etc.)
- Improve existing examples

---

## 📚 Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Models List](https://openrouter.ai/models)
- [Vercel AI SDK](https://sdk.vercel.ai/docs) (TypeScript)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Claude Cookbook - Classification](https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/classification)

---

## 📝 License

This project is for educational and testing purposes.

---

## 🎯 Next Steps

1. **Run the examples** - Start with basic, move to advanced
2. **Try different models** - Compare Claude vs GPT-4 vs Gemini
3. **Test classification** - See RAG and CoT in action
4. **Build something** - Use this as a starting point for your project!

**Happy experimenting!** 🚀
