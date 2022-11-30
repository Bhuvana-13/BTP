import sys
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import statistics
nltk.download('wordnet')
nltk.download('omw-1.4')

target_test = sys.argv[1]	#  Test file argument
target_pred = sys.argv[2]	#  MTed file argument


# Open the test dataset human translation file
with open(target_test,encoding="utf8") as test:
    refs = test.readlines()

#print("Reference 1st sentence:", refs[0])

# Open the translation file by the NMT model
with open(target_pred,encoding="utf8") as pred:
    preds = pred.readlines()

meteor_file = "meteor-" + target_pred + ".txt"
l = []
# Calculate METEOR for each sentence and save the result to a file
for line in zip(refs, preds):
    test = word_tokenize(line[0])
    pred = word_tokenize(line[1])

    meteor = round(meteor_score([test], pred), 2) # list of references
       
    l.append(meteor)
print(statistics.mean(l))