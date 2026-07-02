import pandas as pd
from src.data.dataset import FERdata
from src.model.model import FERModel
from src.training.train import trainModel
import matplotlib.pyplot as plt
data=FERdata()
train,val,test=data.load()
m=FERModel()
model=m.load(activation="relu")
model=m.compile()
history=trainModel(model,train,val,1)

