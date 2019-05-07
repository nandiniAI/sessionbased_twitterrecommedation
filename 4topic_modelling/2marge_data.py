import pandas as pd
df = pd.read_csv('final_slot.csv', error_bad_lines=False)
df1=pd.read_csv('tweet_topic.csv')
topic_num=df1['Dominant_Topic']
topics=[]
for topic in topic_num:
    topic+=1
    topics.append(topic)

print(topics[:5])
df['document_topic']= topics
df.to_csv('topic_allocated_data.csv',index=False)


