import json
import nltk
import sys
from gensim.summarization import summarize
from collections import OrderedDict
import re
SENTENCE_COUNT = 2

# Reads a json file and return a dictionary object


def read_json(file_path):
    temp_json = OrderedDict()
    with open(file_path, 'r') as f:
        temp_json = json.load(f)
    return temp_json
# Write dictionary object to a json formatted file


def write_to_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)
    print("Write into: "+file_path)


def isParagraphValid(paragraph):
    # Check for sentence count. If sentence_count > SENTENCE_COUNT return true.
    return len(nltk.sent_tokenize(paragraph)) > SENTENCE_COUNT


squadTrainingJSON = read_json(sys.argv[1])
squadTrainingJSONData = squadTrainingJSON["data"]

def answerIsIn(summary, qcDict):
    qas = qcDict["qas"]
    for questionsDict in qas:
        if(questionsDict["is_impossible"] == False):
            for answerDict in questionsDict["answers"]:
                answer = answerDict["text"]
                answer_start=answerDict["answer_start"]
                found_in_summary=False
                for m in re.finditer(answer,summary):
                    print(questionsDict)
                    answerDict["answer_start"] = m.start()
                    print('%02d-%02d: %s' % (m.start(), m.end(), m.group(0)))
                    print(answerDict)
                    found_in_summary=True
                    break
                if(found_in_summary == False):
                    questionsDict["is_impossible"] = True
                    break
                print(questionsDict)
# version, data
# data da, [{title, paragraphs}...} var
# paragraphsta [{qas, context}...] var

# tpDict : Title-Paragraph dictionary
# qcDict : Questions - Context dictionary
ind = 0
total = 0
summarized = 0
empty_summarized = 0
skipped = 0
error = 0

ind = 0
for tpDict in squadTrainingJSONData:
    total += len(tpDict["paragraphs"])
    for qcDict in tpDict["paragraphs"]:
        if ind == 5 : exit(1)
        ind+=1
        context = qcDict["context"]
        try:
            summary = summarize(context)
            if summary == "":
                empty_summarized += 1
            else:
                summarized += 1
                qcDict["context"] = summary.replace("\n","").replace("\r","").replace("\t"," ")
                answerIsIn(qcDict["context"],qcDict)
                #print(qcDict["context"])
        except ValueError:
            # Do nothing
            error += 1
print("Successfull Summary Count: " + str(summarized))
print("Empty Summary Count: " + str(empty_summarized))
print("Not Valid(Skipped) Summary Count: " + str(skipped))
print("Total Context Count: " + str(total))
print("Error Count: " + str(error))

write_to_json(squadTrainingJSON, sys.argv[2] + ".json")
