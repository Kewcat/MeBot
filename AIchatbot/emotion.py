import text2emotion as te 
text="I am very happy"
df=te.get_emotion(text)
max_emo= max(df, key=lambda x:df[x])
print(max_emo)


