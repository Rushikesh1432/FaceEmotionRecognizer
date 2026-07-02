from src.model.model import FERModel

def trainModel(model,train,val,epochs):
    return model.fit(
        train,
        epochs=epochs,
        validation_data=val,
        batch_size=32
    )
if __name__=="__main__":
    trainModel()