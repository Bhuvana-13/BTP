import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='te')


# Open the test dataset human translation file and detokenize the references
refs = []

with open("gt.txt",encoding="utf8") as test:
    for line in test: 
        line = line.strip().split() 
        line = md.detokenize(line) 
        refs.append(line)

refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU


# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open("result.txt",encoding="utf8") as pred:  
    for line in pred: 
        line = line.strip().split() 
        line = md.detokenize(line) 
        preds.append(line) 

# Calculate and print the BLEU score
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)
