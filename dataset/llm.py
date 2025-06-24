from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import transformers
import torch
import re
import logging

# Disabilita i log di warning/info di transformers
transformers.logging.set_verbosity_error()
# Disabilita tutti i log di livello INFO
logging.getLogger("transformers").setLevel(logging.ERROR)
class LLM():
    def __init__(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    def rephrase_sentence(self,sentence , max_length=100):
        """
        Riformula una frase usando il modello
        """
        llama_pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, torch_dtype=torch.float16, device_map="auto")
 
        # genera testo
        #output = llama_pipe(f"Rewrite this sentence: '{sentence}' in a natural and realistic way while maintaining the same meaning", max_new_tokens=50)

        # Prompt base
        #caption = "A sea turtle swims near the surface of the ocean."
        
        prompt = f"""
        You are a helpful assistant who rewrites image captions. Your job is to paraphrase the caption provided without altering its main meaning.
        The paraphrased version should be natural, fluent, and grammatically correct. Avoid repeating the exact same structure or wording.
        You can change the sentence or order of the elements to make them more human or stylistically different, but do not generate sentence that are too long.
        Caption:
        "{sentence}"
        
        Paraphrased caption:
        """
        output= llama_pipe(prompt,
                            #messages,
                            #max_new_tokens=25,
                            do_sample=True,
                            temperature=0.7,
                            #top_p=0.9,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

        result = output[0]["generated_text"].split("Paraphrased caption:")[-1].strip()
        
    
        match = re.search(r'Paraphrased caption:\s*"(.*?)"', result, re.DOTALL)
        
        if match:
            result = match.group(1).strip().strip('""')
        else:
            # Fallback se non trova le virgolette
            result = result.split("Paraphrased caption:")[-1].strip()
            # Rimuovi eventuali parti dopo un eventuale nuova riga
            result = result.split('\n')[0].strip().strip('"')
        
        return result
    