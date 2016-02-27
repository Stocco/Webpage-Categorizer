path = "train.txt"
pathtest = "test.txt"
rootlinks=[]
seen=set()
saw={}
import math
import random
import requests
from bs4 import BeautifulSoup
import socket
import struct
import re

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

def farmingvoc(classss):
    vocabulary=""
    for k in classes:
        vocabulary = vocabulary + " " + classes[k]
    x = len(set(vocabulary.split(" ")))
    return x

def categorize(bagofwords):
  bagofwords  = bagofwords.split(" ")
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

def cattest():
  classes,totaldocs,docs = loadtrain(path)
  Voc = float(farmingvoc(classes))
  test = open(pathtest).read().split("\n")
  correct=0
  all=0
  sample = test[random.randint(0,len(test)-1)].split("\t")
  bagofwords = sample[1].split(" ")
  classt = categorize(sample[1])
  all = all + 1
  if(sample[0] == classt[0]):correct = correct + 1
  print("accuracy: ", correct/all)

def striptext(url):
  web = requests.get(url)
  soup = BeautifulSoup(web.text,"html.parser")
  result = soup.findAll("body")
  divs = str(result).split("\n")

  for i in range(0,len(divs)):
      forbiden=0
      divs[i] = re.sub('<[^>]+>','',divs[i])
      if(len(divs[i]) < 110): divs[i] = ''
      forbiden = divs[i].count("{") + divs[i].count("}")
      if forbiden > 1: divs[i] = ''

  texts=""

  for x in divs:
      if(x != ''):
          texts = texts + " " + x

  return texts

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
    fe=["",0]
    times = [[t,rlink.count(t)] for t in seen]
    if(times != []):
        fe = max(times, key=lambda tup: tup[1] )
        saw[fe[0]] = hasval(saw,fe[0]) + 1
    if((rlink in seen) | (hasval(saw,fe[0]) > 40) | (rlink.count(".pdf")) | (rlink.count(".jpg"))):
         return False
    seen.add(rlink)
    try:
      web = requests.get(rlink,allow_redirects=False)
      soup = BeautifulSoup(web.text,"html.parser")
      links = soup.findAll('a')
      for link in links:
          link = link.get('href')
          if(link is not None):
            if(link.count("http://") > 0):
                print("parent: ",link, "webpage")
                extendlink(link)
    except:
        print("Exception")

def hasval(dic,name):
    if(name in dic): return dic[name]
    else: return 0




classes,totaldocs,docs = loadtrain(path)
Voc = float(farmingvoc(classes))
#text,url =striptext("https://www.bba.org.uk/")
#print(text)
#result = categorize(text)
#print(url,"is category: ", result)


scrapper("http://www.mmo-champion.com/content/")
