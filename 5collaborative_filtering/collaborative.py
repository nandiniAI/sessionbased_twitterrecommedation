import pandas as pd
from operator import itemgetter
import numpy as np
import math

df=pd.read_csv('topic_allocated_data.csv')
lines=df.values.tolist()
sorted_lines= sorted(lines, key=itemgetter(3), reverse=False)
col=['created_at','screen_name','text','slot','document_topic']
df=pd.DataFrame(sorted_lines,columns=col)
df.to_csv('Sorted.csv',index=False,columns=col)  


data=pd.read_csv('Sorted.csv')
slots=data['slot']
topics=data['document_topic']
S=max(slots)
r=np.zeros((S+1,11),dtype=int)

l=len(data)
for i in range(l):
    s=data.iloc[i,3]
    t=data.iloc[i,4]
    r[s][t]=1


sim=np.zeros((11,11),dtype=np.float64)
top_pres=np.zeros(11,dtype=np.float64) #stores sqrt(sum(r(s,t)))

for i in range(1,11):
    sum=0
    for j in range(s+1):
        sum=sum+(r[j][i] ** 2)
    top_pres[i]= math.sqrt(sum)
        

for i in range(1,11):
    print(top_pres[i])

for i in range(1,11):
    sim[i][i]=1


for i in range(1,11):
    for j in range(1,11):
        if(i!=j):
            sum=0
            for s in range(S+1):
                if(r[s][i]==1 and r[s][j]==1):
                    sum=sum+1
            sim[i][j]=sum/(top_pres[i] * top_pres[j])

print(len(sim))

for i in range(1,11):
    for j in range(1,11):
        print(sim[i][j], end=' ')
    print('\r')


