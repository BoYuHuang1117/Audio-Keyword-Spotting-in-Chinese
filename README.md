# Audio-Keyword-spotting-in-Chinese
Combine Recurrent Neural Network and Convolutional layer on audio data. The target keyword is "雅婷姊".

***********************************************

# Baseline Model

<img width="910" alt="Capture" src="https://user-images.githubusercontent.com/38172621/146266989-e4757d49-7765-4831-b744-ff644499898e.png">

I. One convolutional layer

II. Two GRU layers

III. Several dropout and batch normalization layer

IV. Fully-connected layer with sigmoid

# Proposed Model

I. One convolutional layer

II. Two GRU layers and modified first into bi-directional and residual

III. Several dropout and batch normalization layer

IV. Fully-connected layer with sigmoid

<img width="910" alt="Capture" src="https://user-images.githubusercontent.com/38172621/146267170-0a4116c4-551a-46fb-8af5-1427213770d7.png">
