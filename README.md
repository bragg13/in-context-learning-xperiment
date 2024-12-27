## How to run

1. Install Ollama:

```bash
# on macOS
brew install ollama

# or visit https://ollama.com/download/ for windows and linux
```

2. Serve Ollama

```bash
ollama serve
```

2. Extend the context length max
3. Download required models from Ollama

```bash
ollama run llama3.1:8b-instruct-q4_0
ollama run mistral:7b-instruct-q4_0
ollama run llama3.2:3b-instruct-q4_0
```

3. Create a virtual environment and install packages

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Run the scripts with desired configurations

```bash
<insert -h here>
```
