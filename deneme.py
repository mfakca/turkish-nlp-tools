import numpy as np
import re
import string
import json
import requests



class Text():
    def __init__(self):
        self.lower_table = str.maketrans("ABCÇDEFGĞHİIJKLMNOÖPRSŞTUÜVYZWXQ","abcçdefgğhiıjklmnoöprsştuüvyzwxq")
        self.noktalama_isaretleri = np.array(["...",".","?","!",":"])

        
    
    def lower(self, text):
        """
        Converts the input to lowercase. (Girdiyi küçük harflere dönüştürür.)

        Example: 
        
        Input:
        
        \ttext [string] => "MERHABA"
        
        Output:
        
        \ttext [string] => "merhaba"
        """
        return text.translate(self.lower_table)
    
    def removePunc(self, text): 
        """
        Removes punctuation in the input. (Girdi içerisindeki noktalama işaretlerini atar.)

        Example: 
        
        Input:
        
        \ttext [string] => "Daha mutlu olamam."
        
        Output:
        
        \ttext [string] => "Daha mutlu olamam"
        """
        
        return text.translate(str.maketrans('', '', string.punctuation))
    
    
    def sentTokenize(self, text):
        """
        Splits input into sentences. (Girdiyi cümlelere böler.)

        Example: 
        
        Input:
        
        \ttext [string] => "Daha mutlu olamam. Bu akşam."
        
        Output:
        
        \tsentences [list <string>] => ["Daha mutlu olamam.", "Bu akşam."]
        """
       
        return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",text)
        
    def wordTokenize(self, text):
        """
        Splits input into words. (Girdiyi kelimelere böler.)

        Example: 
        
        Input:
        
        \ttext [string] => "Daha mutlu olamam. Bu akşam."
        
        Output:
        
        \twords [list <string>] => ["Daha", "mutlu", "olamam", "Bu", "akşam"]
        """
        
        return self.removePunc(text).split()
    
class trainedModel():   
    
    def __init__(self):
        self.SENT_ANALYSIS_API_URL = "https://api-inference.huggingface.co/models/savasy/bert-base-turkish-sentiment-cased"
        self.sent_analysis_headers = {"Authorization": "Bearer api_gxjjPVmHDGvBlYfwPqRkzgecnMbPXjWAaY"}
        self.GENERATE_API_URL = "https://api-inference.huggingface.co/models/adresgezgini/turkish-gpt-2"
        self.generate_headers = {"Authorization": "Bearer api_gxjjPVmHDGvBlYfwPqRkzgecnMbPXjWAaY"}
    
    def sentAnalysis(self,text):
        """
        Calculates the emotion of the input using the trained model. It outputs label and score value. (Eğitilmiş model kullanılarak girdinin duygusunu hesaplar. Çıktı olarak duygu ve skor değerini verir.)

        Trained model: https://huggingface.co/savasy/bert-base-turkish-sentiment-cased
        
        Example: 
        
        Input:
        
        \ttext [string] => "Daha mutlu olamam. Bu akşam."
        
        Output:
        
        \tlabel [string] => "negative" \n
        \tscore [int] => 0.8
        """
        
        data = json.dumps({"inputs":text})
        response = requests.request("POST", self.SENT_ANALYSIS_API_URL, headers=self.sent_analysis_headers, data=data)
        result = json.loads(response.content.decode("utf-8"))[0][0]
        
        return result["label"], result["score"]
        

    def generate(self,text):
        """
        Generates text based on input with a trained GPT-2 model. Returns the produced text as output. (Eğitilmiş GPT-2 modeliyle girdiye dayalı metin üretir. Çıktı olarak üretilmiş metni verir.)

        Trained model: https://huggingface.co/adresgezgini/turkish-gpt-2
        
        Example: 
        
        Input:
        
        \ttext [string] => "Sakince arkana dön bir bak"
        
        Output:
        
        \tlabel [string] => "Sakince arkana dön bir bak ..." 
        """
        
        data = json.dumps({"inputs":text})
        response = requests.request("POST", self.GENERATE_API_URL, headers=self.generate_headers, data=data)
        result = json.loads(response.content.decode("utf-8"))[0]
        
        return result["generated_text"]

   
class Twitter(Text):
    def __init__(self):
        pass
    
    
deneme = Text()
model = trainedModel()
label, score = model.sentAnalysis("Bugün hava güzel ")
print(label)
print(score)
print(deneme.lower("Sakince arkana dön bir bak"))
