# result

|model|epoch|train loss|train accuracy|val loss|val accuracy|learning time|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|model_1|1|0.051|0.980|0.030|0.988|14m27s
|model_2|3|0.064|0.971|0.018|0.993|38m38s
|model_3|3|0.075|0.969|0.022|0.991|48m52s
|model_3|10|0.048|0.980|0.014|0.995|1h24m36s
|model_4|10|0.125|0.946|0.059|0.976|59m29s
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
optimizer : momentumSGD  
scheduler : CosineAnnealingLR  
data augmentation :  
- random flip 
- random crop

## model_3
architecture : resnet50(weights="IMAGENET1K_V2")  
train size : 20000  
validation size : 5000  
optimizer : momentumSGD  
scheduler : CosineAnnealingLR  
data augmentation :  
- random flip 
- random crop
- mixup

## model_4
architecture : alexnet(weights="IMAGENET1K_V1")  
train size : 20000  
validation size : 5000  
optimizer : momentumSGD  
scheduler : CosineAnnealingLR  
data augmentation :  
- random flip 
- random crop
- mixup
