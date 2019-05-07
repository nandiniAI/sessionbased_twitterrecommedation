import pandas as pd
from operator import itemgetter

df=pd.read_csv('topic_allocated_data.csv')
lines=df.values.tolist()
sorted_lines= sorted(lines, key=itemgetter(0), reverse=False)
col=['created_at','screen_name','text','slot','document_topic']
df=pd.DataFrame(sorted_lines,columns=col)
df.to_csv('time_sorted.csv',index=False)  
