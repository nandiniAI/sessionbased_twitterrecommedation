
class Time:
    def __init__(self, hour, minute, second):
        self.hour = hour
        self.minute = minute
        self.second = second

   
    def normalize(self):
        hour = self.hour
        minute = self.minute
        second = self.second
         
        quotient = minute / 60
        if quotient > 0:
            hour += int(quotient)
            minute = int(minute % 60)
            second = int(second % 60)
            
       
        self.hour = hour
        self.minute = minute
        self.second = second
         
        return self
     
    def __add__(self, t):
        """add two times (sum)"""
        hour = self.hour + t.hour
        minute = self.minute + t.minute
        second = self.second + t.second
        res = Time(hour, minute, second)
        res.normalize()
        return res
     
    def __mul__(self, k):
        """multiply a time and an integer constant k (product)"""
        hour = self.hour * k
        minute = self.minute * k
        second = self.second * k
        res = Time(hour, minute, second)
        res.normalize()
        return res
     
    def __lt__(self, t):
        """less than"""
        if self.hour < t.hour or (self.hour == t.hour and self.minute < t.minute and self.second < t.second):
            return True
        else:
            return False
     
    def __eq__(self, t):
        """equal"""
        if self.hour == t.hour and self.minute == t.minute and self.second == t.second:
            return True
        else:
            return False
     
    def __le__(self, t):
        """less or equal"""
        return self < t or self == t
     
    def __gt__(self, t):
        """greater than"""
        return not self <= t
     
    def __ge__(self, t):
        """greater or equal"""
        return self > t or self == t
     
    def __ne__(self, t):
        """not equal"""
        return not self == t
    def __str__(self):
        hour = fill(str(self.hour), 2, '0')
        minute = fill(str(self.minute), 2, '0')
        second = fill(str(self.second), 2, '0')
        return '%s:%s:%s' % (hour, minute, second)



def fill(s, size, c=' ', position='before'):
    """s: string; c: char"""
    if position == 'before':
       s = c * (size - len(s)) + s
    elif position == 'after':
       s += c * (size - len(s))
    return s


#sloting time into sessions
def generate_timeslots(start_time, interval=Time(0,30,0 ), times=48, end_time=None):
    timeslots = []
     
    if end_time is None:
       end_time = start_time + interval*times
    
    time = start_time
    while time < end_time:
          timeslots.append(tuple([time, time + interval]))
          time += interval
     
    return timeslots









import pandas as pd
import csv

csvFile=open("timeslots.csv","w")
Fileout=csv.writer(csvFile, delimiter=',',quoting=csv.QUOTE_ALL)
colnames=['Timestamp', 'slot_num']
Fileout.writerow(colnames)

df = pd.read_csv('process_data.csv')

df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', format="%Y-%m-%d %H:%M:%S")
tweet_time= df['created_at']





start_time = Time(0,0,0)
end_time = Time(24,0,0)
i=1
slot=[]
for start, end in generate_timeslots(start_time, end_time = end_time): 
    slot.append(tuple([start,end,i]))
    i=i+1
    

slots= []
for time_stamp in tweet_time:
    try:
        time_stamp  = pd.to_datetime(time_stamp ,errors='ignore', format='%H:%M:%S').time()
    except ValueError:
            continue
    
    for start, end in generate_timeslots(start_time, end_time = end_time):
        if start<=time_stamp<=end:
           slots.append(tuple([start,time_stamp,end]))

           

for start,time,end in slots:
    for s,e,num in slot:
        if(start==s and end==e):
            slot_num=('%s-%d' %(time,num))
            Fileout.writerow((time,num))






    




