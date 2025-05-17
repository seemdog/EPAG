# EPAG 

Evaluating the Pre-Consultation Ability of LLMs using Diagnostic Guidelines


## Supported Models
`openai`, `claude`, `hf` models 

## Usage

1. Add your keys to `key.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`.
2. Add your custom doctor agent prompt in `./prompt/doctor_prompt.txt`.
3. Run the following script.  

```bash
sh run.sh [model_name] [num_turns] [language] [num_processes] [batch_size] [cut_turns]

```

### Argument Description

- `model_name`: name of the model to be evaluated as the doctor agent that implements pre-consultation(default = gpt-4o-mini).   
- `num_turns`: number of pre-consultation turns(dafaut = 5).   
- `language`: korean or english(default = korean).   
- `num_processes`: number of thread to execute api calls(default = 40).   
- `batch_size`: batch size when evaluating model loaded from huggingface(default = 4).      
- `cut_turns`: (optional) if you wish to use existing simulation dataset of `num_turns` and evaluate the dialogue of less turns(`cut_turns`)(default = `num_turns`). 


