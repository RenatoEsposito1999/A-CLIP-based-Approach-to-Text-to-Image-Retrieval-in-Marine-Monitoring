from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import transformers
import torch
import re
import logging
from transformers import BitsAndBytesConfig
import csv
from tqdm import tqdm
from colorama import Fore, Back, Style




#Disable log
transformers.logging.set_verbosity_error()
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
        Rephrase the sentence using the model
        """
        llama_pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, torch_dtype=torch.float16, device_map="auto")
 
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


        outputs= llama_pipe(messages,
                            max_new_tokens=30,    
                            do_sample=True,    
                            temperature=0.7,    
                            eos_token_id=self.tokenizer.eos_token_id
                            )
        
        result = (outputs[0]["generated_text"][-1])
        return result['content']

def main():
    llm = LLM()
    keys_list = []
    input_file = "Dolphin.csv"
    output_file = "image_captions.csv"
    attempt = 0
    used_list = []
    with open(input_file, 'r', encoding='utf-8') as csvfile, \
        open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        #Initialize the writer and reader
        reader = csv.reader(csvfile)
        writer = csv.DictWriter(outfile, fieldnames=["image_name", "caption"], 
                            quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        next(reader)  # Jump first one 
        keys_list = []
        
        #all the rows inside the csv
        rows = list(reader)
        
        custom_bar = tqdm(rows,
                        desc=f"{Fore.GREEN}🚀 {Back.BLACK}Processing...{Style.RESET_ALL}",
                        bar_format="{l_bar}%s{bar:20}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
                        colour='GREEN',
                        ncols=90,
                        ascii=False,
                        smoothing=0.1)
        
        for row in custom_bar:
            if len(row) >= 2:  # check if there are at least 2 fields
                image_name, caption = row[0], row[1]
                if caption not in keys_list:
                    keys_list.append(caption)
                else:
                    while True:
                        if attempt < 5:
                            print("Call to llm")
                            caption = llm.rephrase_sentence(caption, used_list=used_list)
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
                
                writer.writerow({"image_name": image_name, "caption": caption})

if __name__ == "__main__":
    main()