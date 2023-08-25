import pandas as pd
import numpy as np

vgg=pd.read_csv("submission_vgg.csv")
inception=pd.read_csv("submission_inception.csv")
resnet=pd.read_csv("submission_InceptionResNetV2.csv")
xception=pd.read_csv("submission_xception.csv")
df=pd.DataFrame(columns=["id","vgg","resnet","inception","inception_resnet"])
df["id"]=vgg["id"]
df["vgg"]=vgg["label"]
df["resnet"]=resnet["label"]
df["inception"]=inception["label"]
df["inception_resnet"]=xception["label"]
df['majority'] = df.mode(axis=1)[0].astype(int)
vote=pd.DataFrame()
vote["id"]=df["id"]
vote["label"]=df["majority"]
print(df.tail())
vote.to_csv("majority_voting.csv",index=False)