import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import math
import random
import requests
from bs4 import BeautifulSoup
import re
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
  for i in range(100):
    sample = test[random.randint(0,len(test)-1)].split("\t")
    bagofwords = sample[1].split(" ")
    classt = categorize(sample[1])
    all = all + 1
    if(sample[0] == classt[0]):correct = correct + 1
  print("accuracy: ", correct/all)

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element)):
        return False
    return True

#Core Functions
def farmingvoc(classss):
    vocabulary=""
    for k in classes:
        vocabulary = vocabulary + " " + classes[k]
    x = len(set(vocabulary.split(" ")))
    return x

def categorize(text):
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
         extendlink(rlink)

def extendlink(rlink):
    global seen
    global saw
    global traininglinks
    fe=["",0]
    times = [[t,rlink.count(t)] for t in seen]
    if(times != []):
        fe = max(times, key=lambda tup: tup[1] )
        saw[fe[0]] = hasval(saw,fe[0]) + 1
    if((rlink in seen) | (hasval(saw,fe[0]) > 5) | (rlink.count(".pdf")) | (rlink.count(".jpg")
        |(rlink.count(".tumblr")) | (rlink.count(".exe")))):
         return False

    try:
      web = requests.get(rlink,allow_redirects=False)
      soup = BeautifulSoup(web.text,"html.parser")
      links = soup.findAll('a')
      for link in links:
          link = link.get('href')
          if(link.count("http://") & (link is not None)):
                texto = striptext(link)
                if((texto != '') & (len(texto)>500)):
                    seen.add(link)
                    print(link)
                    traininglinks.append([link,categorize(texto)])
                    extendlink(link)
    except:
        print("Exception")

#Initializing Variables using Training set
classes,totaldocs,docs = loadtrain(path)
Voc = float(farmingvoc(classes))

def main():
  scrapper("http://www.theverge.com/")
  output = open("Links.txt","w")
  output.write("number of links: "+str(len(traininglinks))+"\n")
  for i in traininglinks:
    output.write(str(i[0])+" category: "+str(i[1])+"\n")
  output.close()

main()
