---
layout: post
title: Blog Post 3
---

With the rapid development of technology, we are exposed to large amount of all kinds of information all the time. Our generation knows that how difficult it is to extract effective information because of all the "fake" news. Wouldn't it be nice if we can create an algorithm that helps us detect fake news?

In this Blog Post, we will learn to develop and assess a fake news classifier using *Tensorflow*.

### TensorFlow

*TensorFlow* is designed to help you build models easily. It has a set of APIs that makes it simple to learn and implement machine learning. We can also easily train our models with TensorFlow. 

### Data Source 

Our data for this blog originally comes from the article:

- Ahmed H, Traore I, Saad S. (2017) “Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In: Traore I., Woungang I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer Science, vol 10618. Springer, Cham (pp. 127-138).

This data can be accessed from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). However, the data we are using today has already been cleaned and split into training and testing sets.

### Important Packages

Before we start, here are the packages we will need for today's blog.


```python
import tensorflow as tf
import numpy as np
import re
import string
import pandas as pd

# important tensorflow packages
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

# for splitting training, validation set
from sklearn.model_selection import train_test_split

# for embedding visualization
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
```

##§1. Acquire Training Data 





```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
df = pd.read_csv(train_url)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, there are three important columns in this dataset, `title`, `text`, and `fake`. In the fake column, the data is already encoded to 0 (not fake news) and 1 (fake news), so we don't need to encode this column anymore. 

##§2. Make TensorFlow Datasets

TensorFlow Dataset has a special `Dataset` class that's easy to organize when writing data pipelines.

In this section, we want to write a function called `make_dataset` to construct our `Dataset`that has all the stopwrods removed from `text` and `title` and takes two inputs `text` and `title` of the form `("title", "text")`


```python
# define stopwords 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
```


```python
def make_dataset(df, inputs):
  '''
  this function removes stopwords from desired columns and constructs tensorflow dataset with two input and one output
  input: 
        df - the dataframe of interest
        inputs - a tuple of the form ("title", "text")
  output: a tf.data.Dataset 
  '''
  # remove stopwords from text and title
  df = df[['text','title', "fake"]].apply(lambda x: [item for item in x if item not in stop])
  # construct tf dataset
  # first, construct a dictionary for the inputs 
  d = {}
  for input in inputs:
    d[input] = df[[input]]

  # construct dataset from a tuple of dictionaries
  # the first dictionary is the inputs 
  # the second dictionary specifies the output
  data = tf.data.Dataset.from_tensor_slices((d,{"fake" : df[["fake"]]}))

  # batch the dataset to increase the speed of training
  data = data.batch(100)
  
  return data

```

Now, we use the function we just wrote to construct our `Dataset`.


```python
data = make_dataset(df, ("title", "text"))
```

Next, we'll split the dataset into training and validation sets. We want 20% of the dataset to use for validation.


```python
# shuffle data 
data = data.shuffle(buffer_size = len(data))
```


```python
# 80% of the dataset is for training 
train_size = int(0.8*len(data))

# so we have validation size of 0.2 approximately
train = data.take(train_size)
val = data.skip(train_size)
```


```python
# check the size of training, validation, testing set
len(train), len(val)
```




    (180, 45)



We have 180 batches in training set and 45 batches in validation set, which is exactly 0.8 : 0.2. We have successfully split the data into training and validation sets.

##§3. Create Models

As there are two potential predictors, there are three different potential models: model that focus on only the title of the article, the full text of the article, and both.

Which one is the most effective? 

To address this question, let's create 3 corresponding TensorFlow models.

1. In the first model, we use only the article title as an input.
2. In the second model, we use only the article text as an input.
3. In the third model, we use both the article title and the article text as input.

Compared with Keras sequential API, Keras functioanl API can handle shared layers and multiple inputs, which is exactly what we need for our inquery. For the first two models, we also don’t have 
to create new Datasets. Instead, just specify the inputs to the keras.Model appropriately, and TensorFlow will automatically ignore the unused inputs in the Dataset.


*standardization*

First, we want to standardize text by removing capitals and punctuation.


```python
def standardization(input_data):
  '''
  this function taks a tensorflow dataset as input 
  and convert all text to lowercase and remove punctuation.
  '''
  lowercase = tf.strings.lower(input_data)
  no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
  return no_punctuation 
```

*vectorization*

Next, we want to represent text as a vector. 


```python
# we only want to track the top 2000 distinct words
size_vocabulary = 2000

# vectorization layer learns what words are common 
vectorize_layer = TextVectorization(
    # standardize each sample
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

# adapt the vectorization layer to title and text in the training data
vectorize_layer.adapt(train.map(lambda x, y: x['title']))
vectorize_layer.adapt(train.map(lambda x, y: x['text']))
```

Now that we've prepared our data, it's time to construct our models. The first step is to specify the two inputs using `keras.Input` for our model. 

Note that both `title` and `text` contain just one entry for each news, so the shapes are `(1,)`. 


```python
# inputs

title_input = keras.Input(
    shape = (1,), 
    # name for us to remember for later
    name = "title",
    # type of data contained
    dtype = "string"
)

text_input = keras.Input(
    shape = (1,), 
    name = "text",
    dtype = "string"
)
```

*Hiden Layers*

Let's now write a pipeline for the titles and text. Since `title` and `text` are two different pieces of text but share similar vocabulary, we can use shared layers to encode inputs.


```python
# shared embedding layer
shared_embedding = layers.Embedding(size_vocabulary, 20, name = "embedding")

# pipeline for title
title_features = vectorize_layer(title_input)
title_features = shared_embedding(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)

#pipeline for text
text_features = vectorize_layer(text_input)
text_features = shared_embedding(text_features)
# the dropout rate is higher because lower rate led to overfitting by experiments
text_features = layers.Dropout(0.7)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.5)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)
```

Although it looks like we are performing vectorization on inputs now, the vectorization actually won't take place until we actually run the model.

Now we can specify the respective output layers for three models. For the last model, we will need to `concatenate` the the ouput of `title` pipeline with the output of the `text` pipeline.


```python
# the name of the output layer matches the key corresponding to the target data

# output layer for model1 that only focus on title
title_output = layers.Dense(2,name='fake')(title_features)

# output layer for model2 that only focus on text
text_output = layers.Dense(2,name='fake')(text_features)
```


```python
main = layers.concatenate([title_features, text_features], axis = 1)
main = layers.Dense(32, activation= "relu")(main)

# output layer for model3 that focus on both
output = layers.Dense(2, name = "fake")(main)
```

It's time to create our models! We can do so by specifying the input(s) and output. The `plot_model` function provides an easy way to visually examine the structure of our model, so it's nice to take a look. After compile our model, we can start training it.

### Model 1


```python
# speficy the input and output for model 1
model1 = keras.Model(
    inputs = title_input,
    outputs = title_output
)
```


```python
# check the structure of model 1
keras.utils.plot_model(model1)
```




    
![png](/images/output_32_0.png)
    




```python
# compile the model
model1.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
# fit model 1
history = model1.fit(train, 
                    validation_data=val,
                    epochs = 50)
```

    Epoch 1/50
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py:595: UserWarning:
    
    Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
    
    

    180/180 [==============================] - 2s 10ms/step - loss: 0.6920 - accuracy: 0.5198 - val_loss: 0.6853 - val_accuracy: 0.5233
    Epoch 2/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.6740 - accuracy: 0.6078 - val_loss: 0.5664 - val_accuracy: 0.8676
    Epoch 3/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.5000 - accuracy: 0.8632 - val_loss: 0.3182 - val_accuracy: 0.8978
    Epoch 4/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.3036 - accuracy: 0.8897 - val_loss: 0.2483 - val_accuracy: 0.9093
    Epoch 5/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.2378 - accuracy: 0.9118 - val_loss: 0.2096 - val_accuracy: 0.9238
    Epoch 6/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.2140 - accuracy: 0.9201 - val_loss: 0.1836 - val_accuracy: 0.9316
    Epoch 7/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.2005 - accuracy: 0.9231 - val_loss: 0.1858 - val_accuracy: 0.9278
    Epoch 8/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1833 - accuracy: 0.9268 - val_loss: 0.1932 - val_accuracy: 0.9244
    Epoch 9/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1731 - accuracy: 0.9334 - val_loss: 0.1677 - val_accuracy: 0.9364
    Epoch 10/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1711 - accuracy: 0.9353 - val_loss: 0.1543 - val_accuracy: 0.9451
    Epoch 11/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1635 - accuracy: 0.9375 - val_loss: 0.1528 - val_accuracy: 0.9413
    Epoch 12/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1632 - accuracy: 0.9382 - val_loss: 0.1732 - val_accuracy: 0.9289
    Epoch 13/50
    180/180 [==============================] - 2s 8ms/step - loss: 0.1600 - accuracy: 0.9413 - val_loss: 0.1404 - val_accuracy: 0.9465
    Epoch 14/50
    180/180 [==============================] - 2s 8ms/step - loss: 0.1545 - accuracy: 0.9412 - val_loss: 0.1398 - val_accuracy: 0.9493
    Epoch 15/50
    180/180 [==============================] - 2s 8ms/step - loss: 0.1510 - accuracy: 0.9448 - val_loss: 0.1313 - val_accuracy: 0.9509
    Epoch 16/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1479 - accuracy: 0.9429 - val_loss: 0.1521 - val_accuracy: 0.9431
    Epoch 17/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1455 - accuracy: 0.9447 - val_loss: 0.1341 - val_accuracy: 0.9507
    Epoch 18/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1363 - accuracy: 0.9492 - val_loss: 0.1291 - val_accuracy: 0.9533
    Epoch 19/50
    180/180 [==============================] - 2s 8ms/step - loss: 0.1467 - accuracy: 0.9452 - val_loss: 0.1308 - val_accuracy: 0.9509
    Epoch 20/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1306 - accuracy: 0.9526 - val_loss: 0.1252 - val_accuracy: 0.9537
    Epoch 21/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1403 - accuracy: 0.9465 - val_loss: 0.1418 - val_accuracy: 0.9476
    Epoch 22/50
    180/180 [==============================] - 2s 8ms/step - loss: 0.1349 - accuracy: 0.9490 - val_loss: 0.1292 - val_accuracy: 0.9518
    Epoch 23/50
    180/180 [==============================] - 2s 8ms/step - loss: 0.1348 - accuracy: 0.9488 - val_loss: 0.1225 - val_accuracy: 0.9584
    Epoch 24/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1342 - accuracy: 0.9511 - val_loss: 0.1144 - val_accuracy: 0.9578
    Epoch 25/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1321 - accuracy: 0.9504 - val_loss: 0.1214 - val_accuracy: 0.9551
    Epoch 26/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1270 - accuracy: 0.9530 - val_loss: 0.1430 - val_accuracy: 0.9440
    Epoch 27/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1274 - accuracy: 0.9520 - val_loss: 0.1514 - val_accuracy: 0.9396
    Epoch 28/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1381 - accuracy: 0.9498 - val_loss: 0.1159 - val_accuracy: 0.9573
    Epoch 29/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1226 - accuracy: 0.9551 - val_loss: 0.1179 - val_accuracy: 0.9567
    Epoch 30/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1245 - accuracy: 0.9545 - val_loss: 0.1275 - val_accuracy: 0.9473
    Epoch 31/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1207 - accuracy: 0.9535 - val_loss: 0.1263 - val_accuracy: 0.9502
    Epoch 32/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1191 - accuracy: 0.9559 - val_loss: 0.1243 - val_accuracy: 0.9513
    Epoch 33/50
    180/180 [==============================] - 2s 8ms/step - loss: 0.1231 - accuracy: 0.9549 - val_loss: 0.1118 - val_accuracy: 0.9587
    Epoch 34/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1211 - accuracy: 0.9535 - val_loss: 0.1388 - val_accuracy: 0.9458
    Epoch 35/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1297 - accuracy: 0.9505 - val_loss: 0.1018 - val_accuracy: 0.9618
    Epoch 36/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1173 - accuracy: 0.9564 - val_loss: 0.1104 - val_accuracy: 0.9584
    Epoch 37/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1143 - accuracy: 0.9601 - val_loss: 0.1016 - val_accuracy: 0.9625
    Epoch 38/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1182 - accuracy: 0.9579 - val_loss: 0.1079 - val_accuracy: 0.9589
    Epoch 39/50
    180/180 [==============================] - 2s 8ms/step - loss: 0.1228 - accuracy: 0.9539 - val_loss: 0.1026 - val_accuracy: 0.9620
    Epoch 40/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1178 - accuracy: 0.9553 - val_loss: 0.1179 - val_accuracy: 0.9556
    Epoch 41/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1171 - accuracy: 0.9561 - val_loss: 0.1105 - val_accuracy: 0.9596
    Epoch 42/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1142 - accuracy: 0.9573 - val_loss: 0.1072 - val_accuracy: 0.9600
    Epoch 43/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1196 - accuracy: 0.9549 - val_loss: 0.1036 - val_accuracy: 0.9627
    Epoch 44/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1111 - accuracy: 0.9596 - val_loss: 0.1091 - val_accuracy: 0.9593
    Epoch 45/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1155 - accuracy: 0.9597 - val_loss: 0.1032 - val_accuracy: 0.9611
    Epoch 46/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1121 - accuracy: 0.9574 - val_loss: 0.1057 - val_accuracy: 0.9631
    Epoch 47/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1117 - accuracy: 0.9567 - val_loss: 0.0979 - val_accuracy: 0.9640
    Epoch 48/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1179 - accuracy: 0.9559 - val_loss: 0.1042 - val_accuracy: 0.9631
    Epoch 49/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1107 - accuracy: 0.9600 - val_loss: 0.1063 - val_accuracy: 0.9631
    Epoch 50/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1152 - accuracy: 0.9571 - val_loss: 0.1155 - val_accuracy: 0.9551
    


```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
```




    [<matplotlib.lines.Line2D at 0x7f10c9822b90>]




    
![png](/images/blog3output_35_1.png)
    


Based on the training log and the plogt, we can see that model 1 can reach validation accuracy of around 96%, which is pretty good! 

### Model 2


```python
# specify the input and ouput for model 2
model2 = keras.Model(
    inputs = text_input,
    outputs = text_output
)
```


```python
# take a look at the structure of model 2
keras.utils.plot_model(model2)
```




    
![png](/images/output_39_0.png)
    




```python
# compile model 2
model2.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
# fit model 2
history = model2.fit(train, 
                    validation_data=val,
                    epochs = 50)
```

    Epoch 1/50
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py:595: UserWarning:
    
    Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
    
    

    180/180 [==============================] - 5s 22ms/step - loss: 0.6804 - accuracy: 0.5270 - val_loss: 0.5843 - val_accuracy: 0.6791
    Epoch 2/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.5439 - accuracy: 0.7368 - val_loss: 0.2810 - val_accuracy: 0.9107
    Epoch 3/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.2872 - accuracy: 0.8900 - val_loss: 0.1753 - val_accuracy: 0.9389
    Epoch 4/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.1965 - accuracy: 0.9210 - val_loss: 0.1403 - val_accuracy: 0.9624
    Epoch 5/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.1642 - accuracy: 0.9425 - val_loss: 0.1427 - val_accuracy: 0.9589
    Epoch 6/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.1470 - accuracy: 0.9522 - val_loss: 0.1169 - val_accuracy: 0.9700
    Epoch 7/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.1374 - accuracy: 0.9579 - val_loss: 0.1120 - val_accuracy: 0.9722
    Epoch 8/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.1205 - accuracy: 0.9653 - val_loss: 0.0975 - val_accuracy: 0.9748
    Epoch 9/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.1243 - accuracy: 0.9632 - val_loss: 0.0994 - val_accuracy: 0.9708
    Epoch 10/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.1164 - accuracy: 0.9645 - val_loss: 0.0948 - val_accuracy: 0.9756
    Epoch 11/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.1120 - accuracy: 0.9685 - val_loss: 0.0893 - val_accuracy: 0.9762
    Epoch 12/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.1057 - accuracy: 0.9706 - val_loss: 0.0941 - val_accuracy: 0.9722
    Epoch 13/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0992 - accuracy: 0.9672 - val_loss: 0.0831 - val_accuracy: 0.9744
    Epoch 14/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0945 - accuracy: 0.9690 - val_loss: 0.0694 - val_accuracy: 0.9807
    Epoch 15/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0954 - accuracy: 0.9715 - val_loss: 0.0596 - val_accuracy: 0.9836
    Epoch 16/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0934 - accuracy: 0.9714 - val_loss: 0.0698 - val_accuracy: 0.9809
    Epoch 17/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0824 - accuracy: 0.9752 - val_loss: 0.0659 - val_accuracy: 0.9796
    Epoch 18/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0848 - accuracy: 0.9737 - val_loss: 0.0691 - val_accuracy: 0.9809
    Epoch 19/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0836 - accuracy: 0.9742 - val_loss: 0.0608 - val_accuracy: 0.9840
    Epoch 20/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0752 - accuracy: 0.9755 - val_loss: 0.0544 - val_accuracy: 0.9836
    Epoch 21/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0753 - accuracy: 0.9766 - val_loss: 0.0560 - val_accuracy: 0.9836
    Epoch 22/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0772 - accuracy: 0.9741 - val_loss: 0.0537 - val_accuracy: 0.9842
    Epoch 23/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0743 - accuracy: 0.9780 - val_loss: 0.0502 - val_accuracy: 0.9856
    Epoch 24/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0680 - accuracy: 0.9762 - val_loss: 0.0488 - val_accuracy: 0.9838
    Epoch 25/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0658 - accuracy: 0.9786 - val_loss: 0.0510 - val_accuracy: 0.9856
    Epoch 26/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0727 - accuracy: 0.9770 - val_loss: 0.0496 - val_accuracy: 0.9851
    Epoch 27/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0636 - accuracy: 0.9806 - val_loss: 0.0442 - val_accuracy: 0.9862
    Epoch 28/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0665 - accuracy: 0.9799 - val_loss: 0.0388 - val_accuracy: 0.9863
    Epoch 29/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0652 - accuracy: 0.9772 - val_loss: 0.0401 - val_accuracy: 0.9878
    Epoch 30/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0674 - accuracy: 0.9780 - val_loss: 0.0403 - val_accuracy: 0.9864
    Epoch 31/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0629 - accuracy: 0.9788 - val_loss: 0.0398 - val_accuracy: 0.9884
    Epoch 32/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0518 - accuracy: 0.9824 - val_loss: 0.0420 - val_accuracy: 0.9867
    Epoch 33/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0627 - accuracy: 0.9783 - val_loss: 0.0406 - val_accuracy: 0.9856
    Epoch 34/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0567 - accuracy: 0.9829 - val_loss: 0.0380 - val_accuracy: 0.9891
    Epoch 35/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0525 - accuracy: 0.9839 - val_loss: 0.0326 - val_accuracy: 0.9908
    Epoch 36/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0548 - accuracy: 0.9825 - val_loss: 0.0382 - val_accuracy: 0.9867
    Epoch 37/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0569 - accuracy: 0.9821 - val_loss: 0.0317 - val_accuracy: 0.9901
    Epoch 38/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0559 - accuracy: 0.9811 - val_loss: 0.0409 - val_accuracy: 0.9888
    Epoch 39/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0523 - accuracy: 0.9827 - val_loss: 0.0289 - val_accuracy: 0.9924
    Epoch 40/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0494 - accuracy: 0.9852 - val_loss: 0.0376 - val_accuracy: 0.9890
    Epoch 41/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0539 - accuracy: 0.9836 - val_loss: 0.0361 - val_accuracy: 0.9889
    Epoch 42/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0523 - accuracy: 0.9831 - val_loss: 0.0319 - val_accuracy: 0.9918
    Epoch 43/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0563 - accuracy: 0.9837 - val_loss: 0.0265 - val_accuracy: 0.9915
    Epoch 44/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0494 - accuracy: 0.9833 - val_loss: 0.0295 - val_accuracy: 0.9902
    Epoch 45/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0490 - accuracy: 0.9846 - val_loss: 0.0272 - val_accuracy: 0.9920
    Epoch 46/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0466 - accuracy: 0.9843 - val_loss: 0.0297 - val_accuracy: 0.9927
    Epoch 47/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0433 - accuracy: 0.9853 - val_loss: 0.0304 - val_accuracy: 0.9911
    Epoch 48/50
    180/180 [==============================] - 4s 20ms/step - loss: 0.0473 - accuracy: 0.9863 - val_loss: 0.0312 - val_accuracy: 0.9909
    Epoch 49/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0436 - accuracy: 0.9880 - val_loss: 0.0303 - val_accuracy: 0.9920
    Epoch 50/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0432 - accuracy: 0.9869 - val_loss: 0.0253 - val_accuracy: 0.9922
    


```python
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
```




    [<matplotlib.lines.Line2D at 0x7f1066789210>]




    
![png](/images/output_42_1.png)
    


By reading the training log and the plot, the validation accuracy is able to reach above 98% consistently. This is quite impressive! 

### Model 3


```python
# specify the inputs and output for model 3
model3 = keras.Model(
    inputs = [title_input, text_input],
    outputs = output
)
```


```python
# take a look at the structure of model 3
keras.utils.plot_model(model3)
```




    
![png](/images/output_46_0.png)
    




```python
# compile model 3
model3.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
# train model 3
history = model3.fit(train, 
                    validation_data=val,
                    epochs = 50)
```

    Epoch 1/50
    180/180 [==============================] - 6s 26ms/step - loss: 0.2053 - accuracy: 0.9612 - val_loss: 0.0289 - val_accuracy: 0.9926
    Epoch 2/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0408 - accuracy: 0.9870 - val_loss: 0.0188 - val_accuracy: 0.9942
    Epoch 3/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0303 - accuracy: 0.9905 - val_loss: 0.0195 - val_accuracy: 0.9951
    Epoch 4/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0322 - accuracy: 0.9886 - val_loss: 0.0191 - val_accuracy: 0.9949
    Epoch 5/50
    180/180 [==============================] - 4s 24ms/step - loss: 0.0253 - accuracy: 0.9925 - val_loss: 0.0141 - val_accuracy: 0.9949
    Epoch 6/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0297 - accuracy: 0.9902 - val_loss: 0.0156 - val_accuracy: 0.9936
    Epoch 7/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0248 - accuracy: 0.9926 - val_loss: 0.0123 - val_accuracy: 0.9964
    Epoch 8/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0262 - accuracy: 0.9913 - val_loss: 0.0140 - val_accuracy: 0.9953
    Epoch 9/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0239 - accuracy: 0.9909 - val_loss: 0.0111 - val_accuracy: 0.9969
    Epoch 10/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0212 - accuracy: 0.9925 - val_loss: 0.0097 - val_accuracy: 0.9973
    Epoch 11/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0231 - accuracy: 0.9922 - val_loss: 0.0108 - val_accuracy: 0.9975
    Epoch 12/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0218 - accuracy: 0.9925 - val_loss: 0.0090 - val_accuracy: 0.9980
    Epoch 13/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0253 - accuracy: 0.9923 - val_loss: 0.0131 - val_accuracy: 0.9964
    Epoch 14/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0203 - accuracy: 0.9934 - val_loss: 0.0110 - val_accuracy: 0.9966
    Epoch 15/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0209 - accuracy: 0.9932 - val_loss: 0.0093 - val_accuracy: 0.9973
    Epoch 16/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0208 - accuracy: 0.9928 - val_loss: 0.0068 - val_accuracy: 0.9982
    Epoch 17/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0188 - accuracy: 0.9926 - val_loss: 0.0088 - val_accuracy: 0.9971
    Epoch 18/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0195 - accuracy: 0.9932 - val_loss: 0.0070 - val_accuracy: 0.9980
    Epoch 19/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0176 - accuracy: 0.9943 - val_loss: 0.0073 - val_accuracy: 0.9978
    Epoch 20/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0163 - accuracy: 0.9950 - val_loss: 0.0062 - val_accuracy: 0.9980
    Epoch 21/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0171 - accuracy: 0.9936 - val_loss: 0.0115 - val_accuracy: 0.9964
    Epoch 22/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0213 - accuracy: 0.9921 - val_loss: 0.0066 - val_accuracy: 0.9978
    Epoch 23/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0150 - accuracy: 0.9946 - val_loss: 0.0071 - val_accuracy: 0.9984
    Epoch 24/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0154 - accuracy: 0.9942 - val_loss: 0.0062 - val_accuracy: 0.9991
    Epoch 25/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0175 - accuracy: 0.9941 - val_loss: 0.0045 - val_accuracy: 0.9984
    Epoch 26/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0161 - accuracy: 0.9943 - val_loss: 0.0071 - val_accuracy: 0.9987
    Epoch 27/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0140 - accuracy: 0.9951 - val_loss: 0.0041 - val_accuracy: 0.9991
    Epoch 28/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0164 - accuracy: 0.9948 - val_loss: 0.0099 - val_accuracy: 0.9967
    Epoch 29/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0158 - accuracy: 0.9945 - val_loss: 0.0040 - val_accuracy: 0.9982
    Epoch 30/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0128 - accuracy: 0.9955 - val_loss: 0.0039 - val_accuracy: 0.9987
    Epoch 31/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0107 - accuracy: 0.9964 - val_loss: 0.0037 - val_accuracy: 0.9987
    Epoch 32/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0165 - accuracy: 0.9953 - val_loss: 0.0047 - val_accuracy: 0.9987
    Epoch 33/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0141 - accuracy: 0.9954 - val_loss: 0.0068 - val_accuracy: 0.9978
    Epoch 34/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0130 - accuracy: 0.9956 - val_loss: 0.0095 - val_accuracy: 0.9971
    Epoch 35/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0140 - accuracy: 0.9956 - val_loss: 0.0083 - val_accuracy: 0.9982
    Epoch 36/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0118 - accuracy: 0.9955 - val_loss: 0.0023 - val_accuracy: 0.9993
    Epoch 37/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0147 - accuracy: 0.9947 - val_loss: 0.0089 - val_accuracy: 0.9969
    Epoch 38/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0124 - accuracy: 0.9956 - val_loss: 0.0037 - val_accuracy: 0.9987
    Epoch 39/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0138 - accuracy: 0.9945 - val_loss: 0.0091 - val_accuracy: 0.9976
    Epoch 40/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0126 - accuracy: 0.9966 - val_loss: 0.0039 - val_accuracy: 0.9987
    Epoch 41/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0148 - accuracy: 0.9946 - val_loss: 0.0033 - val_accuracy: 0.9993
    Epoch 42/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0123 - accuracy: 0.9959 - val_loss: 0.0061 - val_accuracy: 0.9984
    Epoch 43/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0156 - accuracy: 0.9941 - val_loss: 0.0033 - val_accuracy: 0.9991
    Epoch 44/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0161 - accuracy: 0.9948 - val_loss: 0.0021 - val_accuracy: 0.9998
    Epoch 45/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0125 - accuracy: 0.9954 - val_loss: 0.0053 - val_accuracy: 0.9993
    Epoch 46/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0109 - accuracy: 0.9963 - val_loss: 0.0029 - val_accuracy: 0.9996
    Epoch 47/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0156 - accuracy: 0.9941 - val_loss: 0.0030 - val_accuracy: 0.9993
    Epoch 48/50
    180/180 [==============================] - 4s 25ms/step - loss: 0.0151 - accuracy: 0.9940 - val_loss: 0.0047 - val_accuracy: 0.9984
    Epoch 49/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0116 - accuracy: 0.9955 - val_loss: 0.0052 - val_accuracy: 0.9982
    Epoch 50/50
    180/180 [==============================] - 5s 25ms/step - loss: 0.0098 - accuracy: 0.9973 - val_loss: 0.0024 - val_accuracy: 0.9993
    


```python
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
```




    [<matplotlib.lines.Line2D at 0x7f106634ecd0>]




    
![png](/images/output_49_1.png)
    


Model 3 is able to consistently reach a validation performance of 99% by the training log and plot. Hence, we pick Model 3 to be our final model, i.e. the model that focuses on both the `text` and `title`. 

##§4. Model Evaluation

From last section, our best model focuses only on the `text`. Now let's test this model's performance on unseen test data. 


```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
test = pd.read_csv(test_url)
```


```python
test = make_dataset(test, ("title", "text"))
```


```python
model3.evaluate(test)
```

    225/225 [==============================] - 4s 16ms/step - loss: 0.0312 - accuracy: 0.9915
    




    [0.03123784437775612, 0.9914918541908264]



The accuracy is 99%! We have created a pretty good fake news detector. 

##§5. Visualizing Embeddings

We can take a step further to learn about which words are learned by our model to be good indicators of fake news by visualizing embeddings.


```python
weights = model3.get_layer('embedding').get_weights()[0] # get the weights from the embedding layer
vocab = vectorize_layer.get_vocabulary()                # get the vocabulary from our data prep for later
```


```python
len(weights[0]) # the dimension of embedding is 20
```




    20




```python
from sklearn.decomposition import PCA
# reduce the dimension to 2
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

# a dataframe of our result 
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
```

Now we are ready to see the plot.


```python
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = list(np.ones(len(embedding_df))),
                 size_max = 2,
                 hover_name = "word")

fig.show()
```





The graph is stretched horizontally and slightly vertically. By hovering the cursor on some of the points, we see `reportedly`, `myanmar`, `rohingya`, `trumps`, `donald`, `barack` are all stronger indicators for whether a news is fake or not. 
