import pandas as pd
from operator import itemgetter
import numpy as np
import math



df=pd.read_csv('time_sorted.csv')
train_data=df[:730]
test_data=df[731:738]
test_slot=151
acess_topic=[]




slots=train_data['slot']

S=max(slots)
print(S)
T=10 #no of topics
r=np.zeros((S+1,T+1),dtype=int)

l=len(train_data)
for i in range(l):
    s=df.iloc[i,3]
    t=df.iloc[i,4]
    r[s][t]=1

for i in range(1,S+1):
    for j in range(1,T+1):
        print(r[i][j], end=' ')
    print('\n')

for i in range(l):
    if (train_data.iloc[i,3]==test_slot):
        acess_topic.append(train_data.iloc[i,4])

acess_topic=set(acess_topic)
print("topic accessed in training set for slot 151", acess_topic)

sim=np.zeros((T+1,T+1),dtype=np.float64)
top_pres=np.zeros(T+1,dtype=np.float64) #stores sqrt(sum(r(s,t)))

for i in range(1,T+1):
    sum=0
    for j in range(S+1):
        sum=sum+(r[j][i] ** 2)
    top_pres[i]= math.sqrt(sum)
        

for i in range(1,T+1):
    print(top_pres[i])

for i in range(1,T+1):
    sim[i][i]=1


for i in range(1,T+1):
    for j in range(1,T+1):
        if(i!=j):
            sum=0
            for s in range(S+1):
                if(r[s][i]==1 and r[s][j]==1):
                    sum=sum+1
            sim[i][j]=sum/(top_pres[i] * top_pres[j])

print(len(sim))

for i in range(1,T+1):
    for j in range(1,T+1):
        print(sim[i][j], end=' ')
    print('\r')
    


no_topic=len(acess_topic)
top_pres=np.zeros((no_topic*5,2),dtype=np.float64)

k=0



for topic in acess_topic:
 flag=np.zeros(T+1,dtype=np.int64)
 for i in range(6):
  large=1
  while(flag[large]!=0):
   large+=1
  
  for j in range(1,T+1):
   if (flag[j]!=1 and sim[topic][j]>sim[topic][large]):
    large=j
  flag[large]=1
  if sim[topic][large]!=1:
   top_pres[k][0]=sim[topic][large]
   top_pres[k][1]=large
   k=k+1
   
print(top_pres)  
freq=np.zeros(T+1,dtype=np.int64)
for topic in top_pres:
    i=int(topic[1])
    freq[i]+=1

'''
maxElement = np.amax(freq)
result = np.where(freq == np.amax(freq))
freq[result]=0
print(freq)
'''
     
length=0
recomTopic= []
while(length!=5):
    maxElement = np.amax(freq)
    for i in range(1,11):
        if(freq[i]==maxElement):
            recomTopic.append(i)
            freq[i]=0
    length=len(recomTopic)



print(test_data)
acess_topic=[]
l=len(test_data)
for i in range(l):
    if (test_data.iloc[i,3]==test_slot):
        acess_topic.append(test_data.iloc[i,4])

print("for slot 151 originally accessed topics in test set are",acess_topic)
print("predicted topics are",recomTopic)
     
