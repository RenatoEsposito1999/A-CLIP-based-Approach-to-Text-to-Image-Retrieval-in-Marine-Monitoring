from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import transformers
import torch
import re
import logging
from transformers import BitsAndBytesConfig
import csv
from tqdm import tqdm
from colorama import Fore, Back, Style




# Disabilita i log di warning/info di transformers
transformers.logging.set_verbosity_error()
# Disabilita tutti i log di livello INFO
logging.getLogger("transformers").setLevel(logging.ERROR)
class LLM():
    def __init__(self) -> None:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,)
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
    
    def rephrase_sentence(self,sentence , max_length=100, used_list=[]):
        """
        Riformula una frase usando il modello
        """
        llama_pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, torch_dtype=torch.float16, device_map="auto")
 
        # genera testo
        #output = llama_pipe(f"Rewrite this sentence: '{sentence}' in a natural and realistic way while maintaining the same meaning", max_new_tokens=50)

        # Prompt base
        #caption = "A sea turtle swims near the surface of the ocean."
        
        prompt = f"""
        You are a helpful assistant who rewrites image captions. Your job is to paraphrase the caption provided without altering its main meaning but correcting factual errors.
        The paraphrased version should be natural, fluent, and grammatically correct. Avoid repeating the exact same structure or wording.
        You can change the sentence or order of the elements to make them more human or stylistically different, but do not generate sentence that are too long.
        Caption:
        "{sentence}"
        
        Paraphrased caption:
        """
        if len(used_list) > 0:

            messages = [
                {"role": "system",
                "content": f"""Rewrite this sentence in a concise way while preserving the original meaning but correcting factual errors.  
                The word "giraffe" is INCORRECT and should be interpreted as "araffe" (a sea turtle) or in general as a sea turtle.  
                If the original sentence mentions "two animals" (two animals, giraffe and or turtles), it is WRONG because the image contains ONLY ONE TURTLE.  
                Ensure the rewritten sentence reflects this correction. 
                Do not copy the original structure.  
                Do not add extra details.  
                Avoid similarity with these previous rephrasings: {used_list} (if provided)."""},
                {"role": "user", "content": f"{sentence}"},
            ]
        else:
            messages = [
                {"role": "system", "content": f"""Rewrite this sentence in a concise way while preserving the original meaning but correcting factual errors.  
                            The word "giraffe" is INCORRECT and should be interpreted as "araffe" (a sea turtle) or in general as a sea turtle.  
                            If the original sentence mentions "two animals" (two animals, giraffe and or turtles), it is WRONG because the image contains ONLY ONE TURTLE.  
                            Ensure the rewritten sentence reflects this correction. 
                            Do not copy the original structure.  
                            Do not add extra details."""},
                {"role": "user", "content": f"{sentence}"},
            ]

        
        
        outputs= llama_pipe(#prompt,
                            messages,
                            max_new_tokens=30,  # Limita la lunghezza    
                            do_sample=True,    
                            temperature=0.7,    
                            eos_token_id=self.tokenizer.eos_token_id
                            )
        '''outputs = llama_pipe(
            messages,
            max_new_tokens=25,
        )'''
        result = (outputs[0]["generated_text"][-1])
        '''result = output[0]["generated_text"].split("Paraphrased caption:")[-1].strip()
        
    
        match = re.search(r'Paraphrased caption:\s*"(.*?)"', result, re.DOTALL)
        
        if match:
            result = match.group(1).strip().strip('""')
        else:
            # Fallback se non trova le virgolette
            result = result.split("Paraphrased caption:")[-1].strip()
            # Rimuovi eventuali parti dopo un eventuale nuova riga
            result = result.split('\n')[0].strip().strip('"')'''
        
        return result['content']

llm = LLM()
keys_list = []
input_file = "/workspace/text-to-image-retrivial/captioner/tmp_image_captions.csv"
output_file = "/workspace/text-to-image-retrivial/captioner/image_captions.csv"
attempt = 0
used_list = []
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'r', newline='', encoding='utf-8') as outfile, \
     open("./definitivo.csv", "w", newline='', encoding='utf-8') as deffile:
    
    # Inizializza reader e writer
    reader_in = csv.reader(infile)
    writer = csv.DictWriter(deffile, fieldnames=["image_name", "caption"], 
                           quoting=csv.QUOTE_ALL)
    writer.writeheader()
    
    reader_out = csv.reader(outfile)
    next(reader_out)
    rows = list(reader_out)
    custom_bar = tqdm(rows,
                        desc=f"{Fore.GREEN}🚀 {Back.BLACK}Processing...{Style.RESET_ALL}",
                        bar_format="{l_bar}%s{bar:20}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
                        colour='GREEN',
                        ncols=90,
                        ascii=False,
                        smoothing=0.1)
    keys_list = []
    for row in custom_bar:
        image_name, caption = row[0], row[1]
        if "giraffe" not in caption:
            keys_list.append(caption)
            writer.writerow({"image_name": image_name, "caption": caption})
            
    
    
    rows_input_file = list(reader_in)
    # Prima converti l'iteratore in lista per conoscere la lunghezza
    
    
    
    # per ogni riga in otupuit file
    for row in custom_bar:
        image_name, caption = row[0], row[1]
        if "giraffe" in caption:
            # cerca image name in input file
            target = image_name
            for row_in in rows_input_file:
                image_name_in, caption_in = row_in[0], row_in[1]
                if target == image_name_in:
                    # RIFRASARE 
                    while True:
                        if attempt < 5:
                            caption = llm.rephrase_sentence(caption_in, used_list=used_list)
                            attempt+=1
                            if caption not in keys_list:
                                keys_list.append(caption)
                                attempt=0
                                used_list = []
                                break
                            else: 
                                used_list.append(caption)
                        else:
                            caption = caption + "_TO_CHECK"
                            attempt= 0
                            used_list = []
                            break
            writer.writerow({"image_name": image_name, "caption": caption})  # Nota: writerow n