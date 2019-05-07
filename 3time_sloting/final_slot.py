import pandas as pd
df1=pd.read_csv('timeslots.csv')
#print(df1[:5])
slot_num=df1['slot_num']
df= pd.read_csv('process_data.csv')
df['slot']= slot_num
df.to_csv('final_slot.csv')

slot_num=df['slot']
screen_name=df['screen_name']

final_slot=[]
for (slot,name) in zip(slot_num,screen_name):
    slot=str(slot)+name
    final_slot.append(slot)


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
final_slot[:] = labelencoder.fit_transform(final_slot[:])

df['slot']=final_slot
df.to_csv('final_slot.csv')

            
