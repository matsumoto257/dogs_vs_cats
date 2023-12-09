# result

|model|epoch|train loss|train accuracy|val loss|val accuracy|learning time|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|model_1|1|0.051|0.980|0.030|0.988|14m27s
|model_2|3|0.064|0.971|0.018|0.993|38m38s
||||
||||


## model_1
architecture : resnet50  
train size : 20000  
validation size : 5000  
optimizer : momentumSGD  

## model_2
architecture : resnet50(weights="IMAGENET1K_V2")  
train size : 20000  
validation size : 5000  
optimizer :  SGD  
scheduler : CosineAnnealingLR  
data augmentation :  
- random flip 
- random crop

