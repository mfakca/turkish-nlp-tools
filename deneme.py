import numpy as np
import re
import string


class Text():
    def __init__(self):
        #self.text = text
        self.lower_table = str.maketrans("ABCÇDEFGĞHİIJKLMNOÖPRSŞTUÜVYZWXQ","abcçdefgğhiıjklmnoöprsştuüvyzwxq")
        self.noktalama_isaretleri = np.array(["...",".","?","!",":"])
    
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
    
    
deneme = Text()
print(deneme.sentTokenize("Merhaba, nasılsın.Bugün"))