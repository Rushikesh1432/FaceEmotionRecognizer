import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Input,MaxPool2D,Dense,Flatten

class FERModel:
    def __init__(self):
        self.Model=None
    def load(self,activation="relu"):
        self.Model=Sequential([
            Input(shape=(48,48,1)),
            Conv2D(32,3,activation=activation),
            MaxPool2D(),
            Conv2D(64,3,activation=activation),
            MaxPool2D(),
            Conv2D(128,3,activation=activation),
            MaxPool2D(),
            Flatten(),
            Dense(64,activation=activation),
            Dense(32,activation=activation),
            Dense(7,activation="softmax")
        ])
        return self.Model




    def predict(self,x):
        return self.Model.predict(x)
    
    def compile(self):
        self.Model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return self.Model

if __name__ == "__main__":
    fer = FERModel()
    model = fer.load(activation="relu")
    model = fer.compile()
    model.summary()