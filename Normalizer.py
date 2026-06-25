from whisper.normalizers import EnglishTextNormalizer
import re

normalizer = EnglishTextNormalizer()

def removePunctuation(text):
    text = ''.join(
        ' ' if c in '!@#$%^&*~-+=_\|;:,.?' else c
        for c in text
    )
    return text

def separateNumbersAndText(text):
    text = re.split('(\d+)', text)
    text = ' '.join(text)
    return text

def splitNumbersIntoDigits(text):
    wrds = text.split()
    for word in wrds:
        if word.isnumeric() and not re.search(r'[\u4e00-\u9fff]+', word): #if digit and not chinese character
            dgts = [int(d) for d in word]
            dgts = ' '.join(str(d) for d in dgts)
            x = wrds.index(word)
            wrds[x] = dgts
        
    return ' '.join(wrds)

def removeSpokenSeparators(text):
    wrds = text.split()
    for word in wrds:
        if word.lower() in ['decimal', 'comma', 'point']:
            x = wrds.index(word)
            wrds[x] = ''
        
    return ' '.join(wrds)

def removeCharSet(text, c1, c2): # for removing all text within (and including) a character set (ex.: [TRANSCRIPT] )
    while c1 in text and c2 in text:
        x = text.find(c1)
        y = text.rfind(c2) # Should be the last entry of the closing element ) ] > 
        text = text[0:x] + text[y+1:]
    return text

def removeNonAlphaNum(text): # for removing all non alphanumeric characters (ex.: ! @ # $ % ^ & * ) (AlphanNum.: A-Z, a-z, 0-9)
    for c in text:
        if c.isalnum() == False and c != ' ' :
            x = text.find(c)
            text = text[0:x] + text[x+1:]
    return text

def filterAndNormalize(text):   
    text = removeCharSet(text, '[', ']')
    text = removeCharSet(text, '<', '>')
    text = removeNonAlphaNum(text)
    text = separateNumbersAndText(text)
    text = removeSpokenSeparators(text)
    text = normalizer(text)
    text = normalizer(text)
    text = splitNumbersIntoDigits(text)
    text = text.lower()

    text = normalizer(text)
    text = splitNumbersIntoDigits(text)

    return text

def normalizeOnly(text):
    return normalizer(text)