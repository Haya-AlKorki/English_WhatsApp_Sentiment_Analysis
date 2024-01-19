import pandas as pd
import numpy as np
import re
def preprocess(data):
    pattern = '\d{1,2}[/.]\d{1,2}[/.]\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s[AaPp][Mm]'
    dates = re.findall(pattern, data)
    messages = re.split(pattern, data)[1:]
    df = pd.DataFrame({'text': messages, 'Date': dates})
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
    df['text'] = df['text'].str.replace(']', '')
    df['text'] = df['text'].str.replace('[', '')
    users = []
    messages = []
    for message in df['text']:
        entry = re.split(' ([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])
    df['User'] = users
    df['Text'] = messages
    df.drop(columns=['text'], inplace=True)
    # Extract date
    df['Only_Date'] = df['Date'].dt.date
    # Extract year
    df['Year'] = df['Date'].dt.year
    # Extract month
    df['Month_num'] = df['Date'].dt.month
    # Extract month name
    df['Month'] = df['Date'].dt.month_name()
    # Extract day
    df['Day'] = df['Date'].dt.day
    # Extract day name
    df['Day_name'] = df['Date'].dt.day_name()
    # Extract hour
    df['Hour'] = df['Date'].dt.hour
    # Extract minute
    df['Minute'] = df['Date'].dt.minute
    # Remove entries having user as group_notification
    df = df[df['User'] != 'group_notification']
    return df