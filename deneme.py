import numpy as np
import re
import string
import json
import requests



class Text():
    def __init__(self):
        #self.text = text
        self.lower_table = str.maketrans("ABCÇDEFGĞHİIJKLMNOÖPRSŞTUÜVYZWXQ","abcçdefgğhiıjklmnoöprsştuüvyzwxq")
        self.noktalama_isaretleri = np.array(["...",".","?","!",":"])
        self.SENT_ANALYSIS_API_URL = "https://api-inference.huggingface.co/models/savasy/bert-base-turkish-sentiment-cased"
        self.sent_analysis_headers = {"Authorization": "Bearer api_gxjjPVmHDGvBlYfwPqRkzgecnMbPXjWAaY"}
        self.GENERATE_API_URL = "https://api-inference.huggingface.co/models/adresgezgini/turkish-gpt-2"
        self.generate_headers = {"Authorization": "Bearer api_gxjjPVmHDGvBlYfwPqRkzgecnMbPXjWAaY"}
        
    
    def lower(self, text):
        return text.translate(self.lower_table)
    
    def removePunc(self, text): 
        return text.translate(str.maketrans('', '', string.punctuation))
    
    # Hatalı
    def sentTokenize(self, text):
        # [^\.{2,}\!\?\:]*[\.{2,}\!\?\:]
        return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",text)
        
    def wordTokenize(self, text):
        return self.removePunc(text).split()
    
        
    def sentAnalysis(self,text):
        data = json.dumps({"inputs":text})
        response = requests.request("POST", self.SENT_ANALYSIS_API_URL, headers=self.sent_analysis_headers, data=data)
        result = json.loads(response.content.decode("utf-8"))[0][0]
        return result["label"], result["score"]
        

    def generate(self,text):
        data = json.dumps({"inputs":text})
        response = requests.request("POST", self.GENERATE_API_URL, headers=self.generate_headers, data=data)
        result = json.loads(response.content.decode("utf-8"))
        return result[0]["generated_text"]

   
    
deneme = Text()
print(deneme.generate("Merhaba, nasılsın. Bugün"))
