import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import math
import random
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
tokenizer = RegexpTokenizer(r'\w+')
#globals DONT TOUCH THEM
path = "train.txt"
pathtest = "test.txt"
rootlinks=[]
seen=set()
saw={}
traininglinks=[]


#helper functions
def hasval(dic,name):
    if(name in dic): return dic[name]
    else: return 0

def textFilter(text):
    arrayofWords = tokenizer.tokenize(text)
    filtered_words = [w for w in arrayofWords if not w in stopwords.words('english')]
    return filtered_words

def loadtrain(path):
  classes={}
  documents={}
  totaldocs=0
  f = open(path)
  docs = f.read().split("\n")
  del docs[len(docs)-1]
  for i in docs:
         first = i.split("\t")

         if(first[0] not in documents):documents[first[0]]=1
         else: documents[first[0]] = documents[first[0]] + 1

         if(first[0] not in classes): classes[first[0]] = first[1]
         else:classes[first[0]] = classes[first[0]] + " " + first[1]

  for k in documents:
    totaldocs = totaldocs + documents[k]

  return classes,totaldocs,documents

def AccuracyTest():
  test = open(pathtest).read().split("\n")
  correct=0
  all=0
  for i in range(500):
    sample = test[random.randint(0,len(test)-1)].split("\t")
    classt = categorizeTopics(sample[1])
    all = all + 1
    if(sample[0] == classt[0]):correct = correct + 1
  print("Ours Naive Bayes  Raw Accuracy is : ", correct/all, " using", all, "random samples")

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element)):
        return False
    return True

def binning(string):
    interest={
        "alt.atheism" : 1,
        "comp.graphics" : 3,
        "comp.os.ms-windows.misc" : 2,
        "comp.sys.ibm.pc.hardware" : 2,
        "comp.sys.mac.hardware" : 2,
        "comp.windows.x" : 4,
        "misc.forsale" : 1,
        "rec.autos" : 1,
        "rec.motorcycles" : 1,
        "rec.sport.baseball" : 1,
        "rec.sport.hockey" : 1,
        "sci.crypt" : 4,
        "sci.electronics"	: 4,
        "sci.med"	: 3,
        "sci.space" : 4,
        "soc.religion.christian" : 1,
        "talk.politics.guns" : 1,
        "talk.politics.mideast" : 2,
        "talk.politics.misc" : 2,
        "talk.religion.misc" : 1
    }
    return interest[string]

def LabeltoNum(label):
    interest={
        "alt.atheism" : 0,
        "comp.graphics" : 1,
        "comp.os.ms-windows.misc" : 2,
        "comp.sys.ibm.pc.hardware" : 3,
        "comp.sys.mac.hardware" : 4,
        "comp.windows.x" : 5,
        "misc.forsale" : 6,
        "rec.autos" : 7,
        "rec.motorcycles" : 8,
        "rec.sport.baseball" : 9,
        "rec.sport.hockey" : 10,
        "sci.crypt" : 11,
        "sci.electronics"	: 12,
        "sci.med"	: 13,
        "sci.space" : 14,
        "soc.religion.christian" : 15,
        "talk.politics.guns" : 16,
        "talk.politics.mideast" : 17,
        "talk.politics.misc" : 18,
        "talk.religion.misc" : 19
    }
    return int(interest[label])

def loadInterest(path):
    interestClass={1:0,2:0,3:0,4:0}
    interestDoc={1:"",2:"",3:"",4:""}
    totalDocs=0
    fr = open(path)

    #x[2][2:-2] to extract category label
    for line in fr.readlines():
        print(line)
        link = line.split(" ")
        InterestNumber = binning(link[2][2:-2])
        text = striptext(link[0])
        interestClass[InterestNumber] = interestClass[InterestNumber] + 1
        interestDoc[InterestNumber] = interestDoc[InterestNumber] + " " + text

    for i in interestClass:
        totalDocs = totalDocs + interestClass[i]

    return interestClass,totalDocs,interestDoc


#Core Functions
def farmingvoc(classss):
    vocabulary=""
    for k in classes:
        vocabulary = vocabulary + " " + classes[k]
    x = len(set(vocabulary.split(" ")))
    return x

def categorizeTopics(text):
  bagofwords  = textFilter(text)
  argmax=[]
  for k in classes:
    proba=0.0
    probdoc = docs[k]/totaldocs
    allwords = float(len(classes[k].split(" ")))
    for word in bagofwords:
       proba = proba + math.log((classes[k].count(word) + 1)/(allwords+Voc))
    argmax.append([k,proba + math.log(probdoc)])
  maxi = max(argmax, key=lambda tup: tup[1])
  return maxi

def striptext(url):
  web = requests.get(url)
  soup = BeautifulSoup(web.text,"html.parser")
  body = soup.findAll(text=True)
  divs = list(filter(visible,body))
  bagofWords=""
  for i in divs:
      i = re.sub('<[^>]+>','',i)
      if(len(i.split(" ")) > 50):
          bagofWords = bagofWords + " " + i

  #Add statement to allow only english pages
  return bagofWords

def scrapper(url):
     global seen
     global  rootlinks
     global  saw
     web = requests.get(url)
     soup = BeautifulSoup(web.text,"html.parser")
     links = soup.findAll('a')
     for link in links:
          link = link.get('href')
          if(link is not None):
            if(link.count("http://") > 0):
                rootlinks.append(link)

     for rlink in rootlinks:
         print("===============================!================================================")
         print("ROOT LINK: ",rlink)
         print("================================================================================")
         if(rlink not in seen): extendlink(rlink)

def extendlink(rlink):
    global seen
    global saw
    global traininglinks
    fe=["",0]
    times = [[t,rlink.count(t)] for t in seen]
    if(times != []):
        fe = max(times, key=lambda tup: tup[1] )
        saw[fe[0]] = hasval(saw,fe[0]) + 1
    if((rlink in seen) | (hasval(saw,fe[0]) > 20) | (rlink.count(".pdf")) | (rlink.count(".jpg")
        |(rlink.count(".tumblr")) | (rlink.count(".exe")))):
         return False

    try:
      web = requests.get(rlink,allow_redirects=False)
      soup = BeautifulSoup(web.text,"html.parser")
      links = soup.findAll('a')
      for link in links:
          link = link.get('href')
          if((link.count("http://") | (link.count("https://"))) & (link is not None) & (link not in seen)):
                texto = striptext(link)
                if((texto != '') & (len(texto)>500)):
                    seen.add(link)
                    print(link)
                    traininglinks.append([link,categorizeTopics(texto)])
                    extendlink(link)
    except:
        print("Exception")

def main():
  scrapper("http://www.theverge.com/")
  output = open("Links.txt","w")
  output.write("number of links: "+str(len(traininglinks))+"\n")
  for i in traininglinks:
    output.write(str(i[0])+" category: "+str(i[1])+"\n")
  output.close()

def svmToolkitTrain():
  docs = open(path).read().split("\n")
  docstest = open(pathtest).read().split("\n")
  text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
 ])

  NBclf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
 ])


  class dados(object):
      data = []
      target = []
      target_names=['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
                    'comp.sys.mac.hardware','comp.windows.x','misc.forsale','rec.autos','rec.motorcycles',
                    'rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med',
                    'sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc',
                    'talk.religion.misc']
  count_vect = CountVectorizer()
  train = dados()
  test =dados()
  del docs[len(docs)-1]
  for i in docs:
      doc = i.split("\t")
      train.data.append(doc[1])
      train.target.append(LabeltoNum(doc[0]))

  del docstest[len(docstest)-1]
  for i in docstest:
      doc = i.split("\t")
      test.data.append(doc[1])
      test.target.append(LabeltoNum(doc[0]))

  clf = NBclf.fit(train.data,train.target)

 # x = clf.predict([test.data[15]])
  docs_eval = test.data
  predicted = clf.predict(docs_eval)
  print("Naive Bayes Results with our Train/test Data")
  print("Raw Accuracy Naive Bayes: ", np.mean(predicted == test.target))



  print(metrics.classification_report(test.target, predicted,
      target_names=test.target_names))

  return clf,train.target_names


#Initializing Variables using Training set
classes,totaldocs,docs = loadtrain(path)
Voc = float(farmingvoc(classes))
#main()
#loadInterest("Links.txt")

AccuracyTest()

#svmToolkitTrain()
# scrapped
#  twenty_train = fetch_20newsgroups(subset='train',shuffle='True',random_state=42)



