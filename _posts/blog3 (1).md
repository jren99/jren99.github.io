# **Blog Post 3**

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




    
![png](output_32_0.png)
    




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




    
![png](output_35_1.png)
    


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




    
![png](output_39_0.png)
    




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




    
![png](output_42_1.png)
    


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




    
![png](output_46_0.png)
    




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




    
![png](output_49_1.png)
    


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


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="419c732f-dc67-4f4d-853a-02bb5e04291d" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("419c732f-dc67-4f4d-853a-02bb5e04291d")) {
                    Plotly.newPlot(
                        '419c732f-dc67-4f4d-853a-02bb5e04291d',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>x0=%{x}<br>x1=%{y}<br>size=%{marker.size}", "hovertext": ["", "[UNK]", "the", "to", "of", "a", "and", "in", "that", "on", "s", "for", "is", "he", "said", "it", "with", "trump", "was", "as", "his", "by", "has", "be", "have", "not", "from", "this", "at", "are", "who", "an", "they", "us", "i", "we", "but", "would", "president", "about", "will", "their", "had", "you", "t", "been", "were", "people", "or", "more", "which", "she", "her", "after", "one", "if", "its", "all", "out", "what", "when", "state", "also", "new", "there", "up", "no", "over", "our", "donald", "house", "states", "can", "government", "clinton", "so", "just", "than", "other", "him", "obama", "some", "republican", "could", "told", "united", "into", "do", "like", "against", "because", "white", "them", "campaign", "any", "news", "last", "two", "now", "time", "election", "how", "only", "party", "first", "even", "former", "being", "should", "year", "country", "did", "while", "many", "years", "before", "hillary", "during", "most", "security", "political", "media", "may", "say", "national", "get", "make", "those", "made", "law", "since", "where", "police", "american", "going", "very", "under", "these", "presidential", "republicans", "court", "percent", "back", "democratic", "bill", "between", "support", "administration", "then", "north", "russia", "think", "week", "including", "know", "way", "trumps", "senate", "according", "public", "america", "officials", "vote", "down", "office", "group", "take", "my", "such", "foreign", "re", "world", "right", "military", "called", "federal", "statement", "million", "don", "saying", "washington", "department", "want", "here", "well", "see", "tuesday", "both", "tax", "much", "still", "congress", "another", "russian", "says", "part", "minister", "friday", "day", "china", "wednesday", "your", "thursday", "through", "work", "go", "asked", "women", "policy", "democrats", "own", "2016", "city", "monday", "need", "off", "war", "me", "deal", "next", "secretary", "committee", "americans", "rights", "whether", "help", "three", "black", "official", "general", "why", "around", "show", "korea", "york", "case", "leader", "never", "does", "took", "man", "members", "come", "same", "use", "senator", "meeting", "order", "report", "good", "candidate", "countries", "without", "intelligence", "left", "really", "put", "end", "power", "times", "every", "used", "trade", "attack", "syria", "fbi", "month", "money", "investigation", "top", "reported", "twitter", "information", "already", "iran", "leaders", "fact", "change", "nuclear", "decision", "groups", "justice", "international", "business", "story", "plan", "long", "family", "days", "voters", "far", "conservative", "south", "too", "several", "again", "months", "interview", "place", "clear", "something", "fox", "likely", "director", "health", "call", "however", "came", "speech", "social", "children", "recent", "press", "got", "believe", "must", "agency", "among", "least", "chief", "barack", "program", "immigration", "move", "things", "issue", "reporters", "john", "might", "won", "border", "ve", "sunday", "home", "number", "act", "islamic", "though", "control", "seen", "major", "m", "didn", "matter", "trying", "post", "supporters", "billion", "point", "killed", "actually", "earlier", "great", "sanders", "today", "later", "spokesman", "went", "nation", "executive", "keep", "found", "school", "thing", "yet", "doesn", "working", "muslim", "system", "become", "economic", "past", "real", "win", "away", "give", "let", "added", "set", "look", "january", "big", "attacks", "march", "little", "making", "himself", "july", "until", "four", "legal", "member", "stop", "companies", "comment", "nothing", "free", "democrat", "violence", "issues", "senior", "prime", "ever", "few", "european", "doing", "forces", "2015", "defense", "human", "lawmakers", "nations", "following", "talks", "held", "known", "taking", "head", "opposition", "across", "continue", "person", "illegal", "process", "care", "eu", "cruz", "given", "expected", "company", "local", "sanctions", "governor", "possible", "force", "action", "once", "enough", "better", "job", "legislation", "important", "wall", "june", "high", "woman", "team", "reports", "course", "source", "financial", "un", "night", "having", "wrote", "community", "released", "history", "evidence", "others", "lot", "face", "open", "nominee", "pay", "men", "syrian", "done", "majority", "life", "taken", "gun", "attorney", "refugees", "close", "union", "private", "anyone", "response", "wants", "question", "judge", "run", "20", "special", "supreme", "staff", "iraq", "plans", "email", "conference", "watch", "ago", "ban", "mexico", "10", "further", "second", "november", "1", "budget", "air", "fight", "early", "using", "along", "behind", "gop", "saturday", "despite", "mr", "anything", "agreement", "ryan", "accused", "am", "able", "letter", "crisis", "comments", "lives", "instead", "efforts", "debate", "university", "someone", "find", "future", "role", "less", "within", "best", "calling", "region", "putin", "race", "five", "sure", "current", "full", "weeks", "economy", "council", "death", "visit", "announced", "lead", "sources", "saudi", "name", "comes", "jobs", "coalition", "getting", "global", "sent", "service", "due", "civil", "hard", "december", "live", "event", "coming", "problem", "ll", "effort", "muslims", "congressional", "d", "2014", "students", "young", "israel", "britain", "running", "nearly", "8", "ties", "chairman", "authorities", "elections", "each", "rules", "october", "thousands", "texas", "september", "emails", "facebook", "center", "position", "daily", "talk", "line", "citizens", "candidates", "votes", "army", "criminal", "allow", "15", "politics", "capital", "wanted", "florida", "paul", "relations", "obamacare", "representatives", "claims", "needs", "street", "led", "leave", "turkey", "immediately", "isn", "weapons", "late", "comey", "healthcare", "ruling", "hold", "began", "climate", "tell", "message", "failed", "start", "middle", "cannot", "policies", "rule", "april", "means", "together", "central", "rather", "lost", "latest", "outside", "officers", "ministry", "everyone", "showed", "liberal", "2017", "peace", "different", "based", "whose", "gave", "words", "2", "services", "racist", "bush", "agencies", "bad", "questions", "east", "speaking", "parliament", "thought", "district", "february", "millions", "reform", "immigrants", "august", "read", "conservatives", "access", "hope", "tried", "elected", "concerns", "enforcement", "reason", "ahead", "sexual", "allies", "try", "stand", "release", "list", "decided", "almost", "energy", "fake", "county", "workers", "strong", "spending", "recently", "meet", "germany", "six", "threat", "chinese", "bring", "planned", "george", "charges", "always", "often", "happened", "allowed", "met", "voting", "oil", "protect", "involved", "poll", "idea", "parties", "rally", "organization", "missile", "laws", "morning", "industry", "makes", "cut", "talking", "century", "especially", "received", "situation", "shooting", "kind", "key", "everything", "3", "europe", "allegations", "side", "entire", "provide", "looking", "seems", "30", "freedom", "include", "voted", "movement", "fire", "james", "officer", "hate", "fighting", "realdonaldtrump", "needed", "host", "denied", "large", "j", "calls", "nomination", "bank", "agreed", "funding", "western", "representative", "insurance", "claim", "12", "hearing", "either", "personal", "market", "adding", "room", "vice", "step", "near", "clearly", "although", "west", "small", "data", "old", "hours", "foundation", "area", "shot", "presidency", "actions", "tweet", "terrorist", "polls", "themselves", "address", "potential", "arrested", "true", "worked", "biggest", "british", "return", "confirmed", "11", "decades", "shows", "feel", "forward", "request", "adviser", "spoke", "serious", "documents", "wrong", "crime", "commission", "water", "leading", "5", "terrorism", "pressure", "alleged", "term", "2012", "appeared", "korean", "declined", "myanmar", "hit", "wife", "soon", "moscow", "passed", "interest", "claimed", "california", "truth", "probably", "building", "nov", "dollars", "fired", "turned", "tillerson", "families", "mean", "simply", "review", "david", "relationship", "continued", "cases", "protesters", "result", "article", "travel", "network", "main", "record", "front", "michael", "toward", "below", "paid", "independence", "love", "25", "influence", "included", "leadership", "view", "forced", "food", "signed", "details", "wasn", "primary", "victory", "saw", "issued", "father", "half", "short", "points", "mccain", "posted", "4", "popular", "mark", "brought", "attempt", "pretty", "sign", "final", "became", "previously", "spent", "mike", "turn", "proposed", "bernie", "seeking", "raised", "started", "friends", "board", "lawyer", "clintons", "account", "hand", "longer", "incident", "21st", "pass", "deputy", "helped", "town", "taxes", "college", "child", "arabia", "total", "regional", "guy", "obamas", "level", "2013", "currently", "son", "agenda", "aid", "fund", "increase", "remarks", "firm", "ask", "hundreds", "repeatedly", "whole", "religious", "created", "respond", "website", "18", "push", "violent", "mayor", "third", "ted", "conflict", "reality", "pence", "independent", "fear", "else", "described", "giving", "robert", "constitution", "areas", "merkel", "protest", "criticized", "protests", "largest", "ambassador", "mass", "isis", "dont", "absolutely", "published", "heard", "al", "armed", "san", "rubio", "education", "similar", "example", "remain", "criticism", "cia", "speaker", "phone", "experts", "convention", "secret", "refugee", "goes", "apparently", "debt", "telling", "cost", "programs", "living", "flynn", "employees", "sessions", "changes", "lower", "iraqi", "build", "base", "hands", "speak", "seven", "stay", "single", "militants", "online", "johnson", "understand", "inside", "appears", "victims", "tweeted", "research", "quickly", "japan", "troops", "northern", "form", "24", "voter", "tv", "risk", "cause", "crowd", "mainstream", "happen", "warned", "rate", "discuss", "carolina", "kurdish", "problems", "german", "events", "proposal", "measures", "flag", "respect", "radio", "christian", "coverage", "senators", "focus", "medical", "businesses", "development", "16", "transition", "television", "photo", "joe", "individuals", "asking", "exactly", "completely", "page", "urged", "dangerous", "politicians", "interests", "opinion", "trip", "spokeswoman", "internet", "served", "charged", "measure", "presidentelect", "100", "share", "presidents", "committed", "seek", "numbers", "prison", "nato", "consider", "13", "moore", "funds", "concern", "southern", "results", "previous", "concerned", "project", "land", "false", "island", "attention", "safety", "certainly", "ground", "brexit", "moment", "france", "powerful", "itself", "drug", "safe", "died", "named", "critical", "50", "poor", "affairs", "14", "reporter", "kelly", "o", "corruption", "church", "responsible", "ready", "gets", "backed", "diplomatic", "book", "rep", "prevent", "fraud", "fellow", "considered", "governments", "whom", "schools", "organizations", "create", "certain", "7", "king", "referendum", "threats", "residents", "provided", "knows", "massive", "expressed", "answer", "yes", "leaving", "chance", "records", "operations", "democracy", "society", "student", "responded", "parents", "choice", "panel", "knew", "hear", "ensure", "6", "repeal", "target", "holding", "terror", "assault", "charge", "difficult", "sides", "cities", "believed", "investment", "french", "expect", "threatened", "series", "terrorists", "growing", "terms", "play", "husband", "filed", "rohingya", "cuts", "9", "critics", "rhetoric", "standing", "approved", "suggested", "takes", "population", "london", "negotiations", "direct", "virginia", "refused", "views", "radical", "mcconnell", "amendment", "behavior", "huge", "class", "parts", "offered", "beijing", "17", "low", "beyond", "paris", "xi", "send", "protection", "maybe", "eight", "car", "cabinet", "favor", "activists", "reached", "agree", "strategy", "gas", "defend", "impact", "mother", "red", "labor", "27", "worst", "sean", "progress", "body", "per", "authority", "28", "newspaper", "finally", "complete", "ordered", "caused", "2011", "star", "sought", "offer", "exchange", "continues", "word", "serve", "attacked", "dead", "crimes", "related", "ways", "killing", "screen", "reach", "individual", "chris", "weekend", "believes", "arms", "regulations", "god", "common", "domestic", "ability", "avoid", "additional", "remains", "willing", "counsel", "iranian", "chicago", "22", "corporate", "rest", "jan", "agents", "2018", "abortion", "courts", "afghanistan", "statements", "regarding", "perhaps", "lack", "effect", "includes", "citing", "raise", "inc", "cover", "19", "supported", "publicly", "opportunity", "facts", "2010", "w", "operation", "period", "macron", "becoming", "test", "status", "promised", "daughter", "joint", "supporting", "looks", "deep", "summit", "gone", "michigan", "sea", "particularly", "multiple", "join", "directly", "sen", "worse", "turkish", "accept", "opposed", "growth", "significant", "replace", "scandal", "spicer", "referring", "credit", "capture", "establishment", "followed", "environmental", "2008", "worth", "gay", "mexican", "costs", "appear", "thinks", "showing", "jr", "considering", "journalists", "block", "arrest", "fiscal", "socalled", "wouldn", "lose", "buy", "regime", "guns", "migrants", "sense", "rival", "friend", "declared", "sept", "eastern", "decide", "21", "towards", "jerusalem", "finance", "electoral", "battle", "announcement", "noted", "lies", "rise", "putting", "mind", "arab", "quite", "higher", "accusations", "canada", "propaganda", "mostly", "minority", "dnc", "supporter", "ohio", "necessary", "trial", "lawyers", "lawsuit", "jeff", "cyber", "stage", "cooperation", "steve", "responsibility", "kids", "trust", "oct", "legislative", "launched", "seem", "green", "begin", "reporting", "kim", "fair", "challenge", "shut", "remember", "dc", "upon", "revealed", "cuba", "vladimir", "income", "puerto", "ended", "communities", "bannon", "price", "tough", "approval", "seriously", "couldn", "association", "subject", "probe", "meetings", "discussed", "conspiracy", "powers", "hollywood", "ally", "language", "manager", "bureau", "briefing", "blame", "approach", "23", "pyongyang", "soldiers", "investigating", "communications", "winning", "alliance", "aimed", "tweets", "classified", "scheduled", "largely", "helping", "racism", "joined", "required", "positions", "yemen", "moving", "homeland", "couple", "accounts", "above", "facing", "banks", "tensions", "meant", "various", "follow", "deals", "de", "technology", "ran", "libya", "site", "infrastructure", "caught", "alabama", "emergency", "controversial", "separate", "israeli", "cast", "available", "goal", "break", "26", "professor", "nbc", "light", "supposed", "ruled", "constitutional", "29", "guilty", "amid", "rich", "ukraine", "reasons", "pushed", "heart", "estate", "conversation", "scott", "pm", "experience", "acting", "resolution", "shared", "promise", "embassy", "stated", "rejected", "investigations", "fully", "version", "borders", "figure", "pick", "african", "coal", "jones", "hill", "demand", "bit", "aren", "values", "russians", "murder", "game", "average", "removed", "decisions", "carry", "broke", "uk", "transgender", "hurt", "billionaire", "aides", "addition", "60", "property", "im", "islam", "iowa", "appeal", "sex", "present", "played", "kill", "alone", "voice", "mueller", "familiar", "changed", "rightwing", "prosecutors", "doubt", "2009", "felt", "amount", "taiwan", "lie", "piece", "damage", "reduce", "throughout", "lying", "moved", "hannity", "allowing", "stories", "pointed", "mission", "claiming", "warning", "steps", "msnbc", "angry", "opponents", "solution", "explain", "40", "stopped", "province", "literally", "judges", "jail", "hopes", "career", "appeals", "carried", "benefits", "faced", "annual", "nor", "highly", "abc", "targeted", "warren", "territory", "seat", "identified", "please", "born", "beginning", "focused", "delegates", "analysis", "written", "matters", "dropped", "paying", "happy", "ceo", "carson", "designed", "wikileaks", "treasury", "fall", "condition", "argued", "strike", "increased", "compared", "asia", "playing", "cited", "works", "brown", "resources", "martin", "hospital", "allegedly", "internal", "bangladesh", "australia", "partners", "conditions", "sentence", "conduct", "starting", "server", "africa", "sort", "positive", "institute", "airport", "activities", "treatment", "romney", "pentagon", "pushing", "particular", "lady", "judicial", "featured", "lebanon", "none", "aide", "islamist", "arrived", "whatever", "totally", "losing", "investors", "inauguration", "victim", "partner", "collusion", "suspected", "surprise", "bloc", "veterans", "unless", "receive", "militant", "bomb", "arizona", "affordable", "require", "planning", "minutes", "loss", "decade", "bid", "bringing", "secure", "possibly", "female", "closed", "assembly", "erdogan", "requests", "possibility", "age", "thats", "spain", "frontrunner", "drew", "picture", "management", "basis", "admitted", "girl", "save", "note", "sales", "religion", "hotel", "campaigns", "restrictions", "detained", "defeat", "streets", "negative", "involvement", "illegally", "remove", "humanitarian", "benefit", "specific", "prior", "audience", "rico", "ongoing", "mattis", "discussion", "intended", "hell", "faces", "sometimes", "prepared", "ones", "kept", "park", "markets", "judiciary", "document", "scene", "leftist", "2017realdonaldtrump", "prosecutor", "nobody", "assad", "opened", "numerous", "biden", "behalf", "abuse", "reforms", "systems", "kremlin", "flint", "field", "testimony", "payments", "nine", "marriage", "jim", "increasingly", "explained", "offensive", "built", "zone", "study", "hacking", "detroit", "highest", "confirmation", "seats", "entering", "dozens", "thank", "reportedly", "immediate", "greater", "bills", "ben", "wait", "ultimately", "parenthood", "vowed", "ethnic", "communist", "clean", "eventually", "conducted", "names", "missiles", "miles", "thanks", "sarah", "eric", "dec", "brussels", "backing", "ad", "levels", "keeping", "check", "associated", "31", "wisconsin", "training", "21wire", "civilians", "attempted", "worried", "otherwise", "happening", "evening", "conway", "watching", "politically", "players", "vietnam", "stance", "seemed", "resignation", "prices", "failure", "diplomats", "alternative", "activist", "standards", "fighters", "dr", "tom", "suspect", "oh", "blocked", "venezuela", "racial", "estimated", "disaster", "choose", "blamed", "appointed", "happens", "fuel", "drive", "jersey", "progressive", "getty", "boost", "spend", "raising", "herself", "citizen", "advance", "strikes", "attend", "stood", "richard", "retired", "prove", "listen", "date", "correct", "treated", "sheriff", "innocent", "grand", "dismissed", "aware", "extremely", "wounded", "republic", "paper", "natural", "manafort", "limited", "easy", "confidence", "campus", "broadcast", "liberals", "gives", "figures", "cash", "truly", "thinking", "strategic", "institutions", "destroy", "capitol", "administrations", "investigators", "vehicle", "potentially", "lee", "christie", "wonder", "schumer", "failing", "agent", "tells", "signs", "road", "production", "platform", "pakistan", "linked", "door", "ballot", "opening", "kushner", "commitment", "asylum", "sitting", "rates", "path", "proof", "involving", "chair", "actual", "narrative", "denies", "contact", "sending", "resign", "investigate", "culture", "attended", "providing", "fine", "faith", "donors", "dialogue", "serving", "promote", "polling", "palestinian", "housing", "hour", "guard", "search", "mention", "veteran", "tehran", "neither", "navy", "looked", "assistance", "yesterday", "pledged", "pennsylvania", "hes", "shown", "aliens", "screenshot", "activity", "injured", "floor", "club", "waiting", "smith", "lines", "ireland", "drop", "unlikely", "limit", "seeing", "links", "attempts", "35", "surveillance", "prince", "homes", "epa", "code", "carrying", "ballistic", "strongly", "minimum", "interior", "grant", "simple", "heads", "hall", "convicted", "camp", "widely", "wealthy", "successful", "ross", "range", "coup", "congressman", "christmas", "space", "products", "overseas", "orders", "heavily", "benghazi", "appearance", "taxpayers", "st"], "legendgroup": "", "marker": {"color": "#636efa", "size": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "sizemode": "area", "sizeref": 0.25, "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scattergl", "x": [0.27352356910705566, -0.20031416416168213, -0.8135619163513184, 0.2011825293302536, 0.5316191911697388, 0.6624466180801392, -0.008014568127691746, 1.0179122686386108, -1.2266740798950195, 2.938981771469116, -4.687939167022705, 0.4035729467868805, -1.5781586170196533, 0.19737564027309418, 6.277108669281006, -0.7803537249565125, -0.24758002161979675, -0.6970077753067017, -0.6551787257194519, 1.3496853113174438, 0.14864444732666016, -0.023512082174420357, -0.04464304819703102, 0.38346680998802185, 0.002166387625038624, 0.4661407470703125, 0.6358383297920227, -3.7966697216033936, 0.2526456415653229, -1.1437193155288696, -0.1619407683610916, -0.4518294036388397, -0.957408607006073, 2.249237537384033, 2.9683563709259033, 1.6358978748321533, 3.0363683700561523, -0.5918663740158081, 0.6865805983543396, -0.9698082804679871, -0.5226993560791016, -1.1745336055755615, 0.789860725402832, -2.32041072845459, -7.1527557373046875, -1.827009916305542, 1.4107375144958496, 1.2584290504455566, 0.6681244969367981, -1.5147883892059326, 0.2798226475715637, 0.23041805624961853, -0.4828161895275116, 0.21090246737003326, 0.1935528814792633, 1.6940606832504272, 3.7113795280456543, -1.1124749183654785, -1.5764527320861816, -0.48281675577163696, -0.04286845028400421, 0.13112379610538483, -0.4163696765899658, 0.43285277485847473, 0.55854332447052, -0.05958773195743561, -0.23253600299358368, 1.438267707824707, -3.0049800872802734, 0.025234587490558624, 1.401601791381836, 1.9284603595733643, -0.9054228663444519, 2.082371234893799, -0.21534502506256104, 0.02635367028415203, -5.52313232421875, 0.6829141974449158, 0.0010378399165347219, -1.231958031654358, -3.713710069656372, 3.6455037593841553, 2.0364530086517334, 0.8089501857757568, 3.4148809909820557, -0.3577726483345032, -1.0522247552871704, 0.40213295817375183, -2.8367481231689453, -0.3526158034801483, -0.9071140289306641, 0.36982783675193787, -1.107656478881836, 0.9034041166305542, -0.6298362016677856, -2.333090305328369, 1.4271374940872192, 1.1892774105072021, -0.7065700888633728, -0.12729743123054504, 2.8395235538482666, -2.6658143997192383, -1.458594799041748, 2.0898489952087402, 0.7956557273864746, -3.199312686920166, 1.4687609672546387, -1.7329998016357422, -1.3259843587875366, 3.9051132202148438, 2.195297956466675, -0.8478939533233643, 0.026028761640191078, 1.4371721744537354, 2.214646339416504, 0.7770613431930542, -6.523263454437256, -1.0037060976028442, -0.92875736951828, -0.09324228018522263, -0.02864746004343033, 1.3955646753311157, 0.8591445684432983, 3.565187931060791, 0.42880815267562866, 0.641639769077301, 0.1441434621810913, -0.11825715750455856, -0.670634388923645, 0.7814521193504333, 1.0460964441299438, 1.3380260467529297, 1.0549181699752808, -5.412402153015137, -0.6266470551490784, 1.1087685823440552, 1.3951002359390259, -6.177535057067871, 4.062502384185791, -1.2478803396224976, 0.40905430912971497, 2.542572259902954, -1.0060086250305176, 1.6461520195007324, -0.32662433385849, 0.2832454442977905, 0.922173798084259, -0.9496265649795532, -1.0884259939193726, 0.5612949132919312, 0.3790105879306793, 0.30667006969451904, 0.9887247085571289, 0.7466263771057129, 0.6527694463729858, -0.6754339337348938, 14.873620986938477, 2.5308120250701904, -1.8453665971755981, -0.09028010815382004, -5.163995742797852, 0.48398637771606445, 0.009014598093926907, -3.2151694297790527, 1.2635592222213745, 0.9191848039627075, -1.6226122379302979, -0.6084839701652527, -0.16056787967681885, -0.41146326065063477, -6.209916114807129, 0.0018136827275156975, -1.4776179790496826, 0.1520809680223465, -0.05646747350692749, -2.643786668777466, 2.9668211936950684, 1.6482386589050293, -2.2166900634765625, 2.723721742630005, -1.2654508352279663, -2.5765604972839355, 0.24487175047397614, -3.159628391265869, -0.13614977896213531, 2.4626080989837646, 4.875401973724365, -0.2548179626464844, 1.7128583192825317, -1.8871850967407227, 1.0797817707061768, 0.26662686467170715, -2.904193162918091, -1.063903570175171, 1.8385928869247437, 0.4706285297870636, 3.9067413806915283, 3.815959930419922, -1.2941854000091553, 1.8283363580703735, 6.205698013305664, -1.6026583909988403, 4.426022529602051, 0.48976394534111023, 1.3238046169281006, -2.5219264030456543, 1.266705870628357, 1.5085651874542236, 0.40059441328048706, -0.5475585460662842, -0.3013111650943756, -2.8244900703430176, 2.277515411376953, 4.021622657775879, -1.2099368572235107, 0.4014284610748291, -0.12575161457061768, 1.1571335792541504, -0.3801901638507843, 1.4576526880264282, 0.24214626848697662, 1.644544243812561, -1.426565408706665, 0.8160589337348938, 3.8489694595336914, -0.23194527626037598, 0.4671739339828491, -1.9668046236038208, 1.6721117496490479, 0.4778802990913391, -3.611482858657837, 3.3769612312316895, 1.138450264930725, -0.2610892355442047, -0.1582288146018982, 1.9058868885040283, 2.544306516647339, -1.011765956878662, -2.306936740875244, -0.8568295240402222, -0.38243550062179565, -1.5903804302215576, 0.001731271855533123, -0.7085384130477905, 0.25579074025154114, 1.1556357145309448, -1.087704062461853, -0.19610559940338135, -1.6504595279693604, -0.37688860297203064, 1.0750871896743774, 0.5516688823699951, 0.6662184000015259, -1.0495679378509521, -2.812664031982422, -2.441427230834961, -2.363342761993408, 0.9714126586914062, 2.1838321685791016, -1.769697904586792, -3.161611795425415, -0.35989436507225037, 4.681817054748535, -0.9807966947555542, -1.7182610034942627, -2.438450813293457, 2.1301512718200684, -2.9299051761627197, -0.6478184461593628, 1.1273454427719116, 1.1961004734039307, 5.6029486656188965, 1.1579517126083374, -1.0995311737060547, 0.21226011216640472, 0.001539858989417553, -4.356744766235352, 0.4489630460739136, -0.487420916557312, 0.7959588766098022, 1.4998819828033447, 0.7020849585533142, -0.35603758692741394, 0.8464382886886597, -0.4669569730758667, 0.5383297801017761, 0.03512278571724892, 0.5535749197006226, 0.9132221341133118, 1.4080506563186646, 1.775510311126709, -0.9048793911933899, 4.919893741607666, 0.20978280901908875, -3.1002228260040283, -3.507251024246216, 1.3879518508911133, 2.664853811264038, -1.7524969577789307, 1.4440698623657227, -3.5391812324523926, -3.294020175933838, 0.34081223607063293, 0.6398082971572876, 0.5993189215660095, -0.2803592085838318, 0.09812572598457336, -0.776043713092804, -0.704354465007782, 0.7240884304046631, 1.8763874769210815, -2.0539257526397705, -4.204300403594971, -3.9830024242401123, -0.4983164370059967, 0.7193577885627747, 3.7409732341766357, 2.1558876037597656, 0.6648856997489929, 3.1601169109344482, 0.005208779126405716, 0.3794308602809906, 0.7951080203056335, -1.128629207611084, -1.2588528394699097, 0.541327714920044, 3.614737033843994, -5.128706455230713, -1.8217370510101318, -0.5585314631462097, 1.0052913427352905, -4.392651557922363, 0.29377588629722595, 1.785703182220459, -2.276437759399414, -1.2371318340301514, 3.333665609359741, 0.3080618679523468, 1.4549803733825684, 2.880192756652832, -1.3811194896697998, -5.054777145385742, -3.5742859840393066, -2.6645588874816895, 0.6251614689826965, -0.7689668536186218, -2.5467417240142822, 0.7395586967468262, -3.6420862674713135, 4.5165276527404785, -4.277573108673096, 2.0770654678344727, 0.08316086232662201, -0.9638306498527527, -1.6193434000015259, 3.4012603759765625, 4.980100154876709, -4.393779277801514, -1.268067717552185, -1.6533757448196411, 0.6044382452964783, -0.026531679555773735, -0.8620075583457947, -1.582471489906311, -1.126057505607605, -4.541815757751465, -0.4919240176677704, -5.519176006317139, 0.9854477643966675, -1.8025407791137695, -0.35711321234703064, -0.8923867344856262, -2.6222174167633057, -0.7005484104156494, -0.2044464647769928, -0.7797036170959473, -1.870104193687439, 1.1468236446380615, -0.01028958335518837, -0.5340099930763245, -1.7393693923950195, -0.173701673746109, 1.1745284795761108, -1.286909818649292, 0.7103557586669922, -0.4537845849990845, -2.2308335304260254, 0.04144863411784172, 0.5810744762420654, 1.2503628730773926, 1.169611930847168, -2.7147247791290283, -0.8492370247840881, 2.41848087310791, 4.047242164611816, -1.9958301782608032, -0.21594995260238647, -1.309004306793213, -0.4302273094654083, -1.6642427444458008, 1.2260417938232422, -0.038526684045791626, -1.2072142362594604, 0.4416795074939728, 1.9660062789916992, -0.612446665763855, 1.5805500745773315, -1.049027919769287, 0.5145700573921204, -0.9924451112747192, 2.971574068069458, 1.0740349292755127, 2.4661900997161865, 5.692946434020996, 1.5624868869781494, 0.46016794443130493, -2.1440510749816895, 3.2249011993408203, 6.464018821716309, -0.5382238030433655, -1.3523222208023071, -1.8123924732208252, -3.118858814239502, -0.44742873311042786, -2.9571683406829834, 1.1084550619125366, 1.5636166334152222, -0.8869604468345642, 2.3380801677703857, 0.575745165348053, 1.8729506731033325, -0.7514384984970093, 0.01671799272298813, 0.5138511657714844, -1.64366614818573, -0.1943376213312149, 1.1909246444702148, -0.17579302191734314, -0.24304206669330597, 0.7315827012062073, 1.3487722873687744, 0.4224037826061249, -1.7256406545639038, -0.30892160534858704, 0.32368627190589905, -1.5985126495361328, -0.13849924504756927, -2.5322346687316895, -0.7463148832321167, 5.061960697174072, -0.7272207140922546, 1.4043422937393188, -3.632120132446289, -3.4709692001342773, 2.1943445205688477, -0.7587549686431885, -1.2505207061767578, 1.224823236465454, -2.8282761573791504, 3.0757064819335938, 0.4353265166282654, -0.04412510618567467, -0.5091643333435059, 0.01786644384264946, -2.5452041625976562, 0.3546065390110016, -0.16723763942718506, 0.2802312672138214, -0.20092318952083588, -0.45220059156417847, -0.4334602952003479, -1.5624752044677734, 0.8637305498123169, -2.249952793121338, 1.2318035364151, 2.20583176612854, 0.062489960342645645, -7.031770706176758, -2.242281913757324, 0.0692974105477333, -1.429079294204712, -1.3338515758514404, 0.8475634455680847, -2.1780853271484375, -0.1648360788822174, -2.618359088897705, 1.3362362384796143, -0.2597988247871399, -1.2841286659240723, -0.2700880169868469, 0.9560757875442505, -9.211016654968262, 0.16347001492977142, -0.025778625160455704, 0.8246212601661682, -1.159123420715332, 1.3261975049972534, 1.9332518577575684, 1.3396228551864624, -1.4962464570999146, 1.3125637769699097, 2.0306661128997803, 1.2841185331344604, 0.9725107550621033, -0.7799829840660095, 0.1893957257270813, 0.7938363552093506, -13.713693618774414, 2.7253341674804688, 0.662841796875, -8.17410945892334, -3.1248953342437744, 0.6805506944656372, -0.019810814410448074, 2.2402353286743164, 0.7089622020721436, 0.773137092590332, 0.10006749629974365, 2.4614930152893066, 1.794616460800171, -2.9241769313812256, -2.665071725845337, 0.20696522295475006, 0.9440031051635742, 1.4281103610992432, -2.0018470287323, -0.04840962588787079, 1.7455378770828247, 2.5281941890716553, -0.4289698004722595, -0.6377512216567993, -1.5513367652893066, -2.0193638801574707, 5.037946701049805, 0.2863755524158478, 1.9045175313949585, 1.1145045757293701, -3.3858864307403564, -1.281545639038086, -1.7594859600067139, -0.4567457139492035, 1.8373546600341797, -1.3363418579101562, 1.3595945835113525, 1.3107531070709229, -2.5850563049316406, 2.064419984817505, 2.2522380352020264, -0.34874624013900757, 0.2180308699607849, -5.929743766784668, -0.9077109098434448, 3.5185587406158447, -1.1540015935897827, -0.4773840308189392, -0.2894822061061859, 2.2781810760498047, 1.4679679870605469, 0.4991854727268219, 2.5095674991607666, 0.649355947971344, -3.1720149517059326, -0.5147896409034729, -0.3783169090747833, 0.64895099401474, -2.969660520553589, -3.0626654624938965, -1.6618953943252563, -0.3696296513080597, -0.4520803391933441, -0.5113760232925415, -0.33642128109931946, 0.969868004322052, 0.942020058631897, 5.993524074554443, -2.6366989612579346, 0.7119162678718567, 0.4651651084423065, -0.5410866737365723, 0.17076683044433594, 1.773497462272644, -1.073533296585083, 2.2622363567352295, 1.495192050933838, -1.5585002899169922, 0.4966074228286743, -1.5779861211776733, 1.7989139556884766, -1.241803526878357, -1.8834664821624756, 1.8514280319213867, -1.8553955554962158, -6.191191673278809, 2.3859424591064453, -2.729952335357666, -3.1671862602233887, 1.315717101097107, -0.6393861770629883, 1.7782655954360962, -3.7570338249206543, -1.7102984189987183, 0.4997636675834656, 1.5093921422958374, 6.59589958190918, -0.8620129227638245, 1.7779704332351685, -2.689328670501709, -1.1873935461044312, 0.41478878259658813, 4.6199564933776855, -1.7891054153442383, -1.5828750133514404, -1.0788394212722778, 2.21050763130188, -0.707537055015564, 5.23261022567749, 0.4358140528202057, -5.181659698486328, 0.23727987706661224, 2.8109142780303955, 0.787846565246582, 4.489348888397217, 5.59283447265625, 1.0825926065444946, -0.5368351936340332, -1.5049762725830078, 0.0619296170771122, -1.1881681680679321, -0.6761484742164612, 0.20256079733371735, -1.987587571144104, 0.5413355827331543, 3.808678150177002, 2.500129461288452, -2.774125576019287, 1.9675123691558838, 2.810488700866699, 2.820984363555908, -1.658401370048523, 0.5568938851356506, -1.4654885530471802, 0.8538159728050232, -3.2536416053771973, 5.570064544677734, -2.212594509124756, 3.5247039794921875, -5.278651714324951, -4.643107891082764, 2.661874532699585, 3.2544705867767334, -1.7603451013565063, 0.4581509828567505, -2.18223237991333, -4.144427299499512, -2.117631673812866, 0.5132155418395996, -6.358616352081299, -2.6408066749572754, 3.243556022644043, -1.027991771697998, 0.9948301911354065, 1.7976008653640747, 0.23424546420574188, 8.358888626098633, -3.2492563724517822, 1.2399529218673706, -1.6630653142929077, -2.811298370361328, 1.1042147874832153, 0.2519686222076416, 1.6772425174713135, -7.185567378997803, 0.46809208393096924, 0.23350419104099274, 2.8447484970092773, -1.1674522161483765, -1.5663036108016968, 1.9582443237304688, 0.5750538110733032, -4.0029425621032715, 0.7990276217460632, -1.406798243522644, 2.084899425506592, 1.5355530977249146, -3.522456407546997, -1.461069107055664, -2.037346363067627, -0.45993033051490784, -0.7836387157440186, -0.5148193836212158, 0.858485996723175, -5.320801734924316, -0.3387524485588074, 2.6235525608062744, -0.021855928003787994, -4.271517753601074, 1.5116719007492065, 0.23527368903160095, 0.6838447451591492, 0.26075443625450134, 0.021054262295365334, -1.793333649635315, -2.3640055656433105, -0.5864711403846741, 1.5072904825210571, 1.813927412033081, 3.7733142375946045, -2.019684076309204, -0.9569790363311768, -2.005565643310547, -1.1909947395324707, 3.0175726413726807, 0.6707209944725037, -3.2924323081970215, 1.3449034690856934, -1.5084515810012817, 3.3825395107269287, -0.3073277473449707, 1.9784210920333862, -1.0929359197616577, 0.6815184354782104, -5.261469841003418, 1.6516146659851074, -2.8417439460754395, -0.488179475069046, -2.4730777740478516, -4.4672112464904785, 0.5620002150535583, -3.337157726287842, 2.3500239849090576, -3.645477771759033, -3.993269443511963, -0.04781604930758476, -3.014209032058716, -1.9983173608779907, -1.0218446254730225, 2.3491923809051514, -2.0080177783966064, -5.335641384124756, 1.9755375385284424, 1.722893238067627, -6.278963565826416, -1.5932649374008179, -1.203321099281311, -1.0459787845611572, -4.166691303253174, 0.5644026398658752, 0.5536483526229858, -2.6827423572540283, -3.1479015350341797, -4.359830856323242, 2.3771815299987793, 4.4677042961120605, 1.7287485599517822, -3.185612440109253, 3.5886027812957764, 2.6725993156433105, -6.342523097991943, 3.016343355178833, 2.6404988765716553, 3.53741455078125, 2.1469101905822754, 1.557498812675476, 0.21561811864376068, 3.0414459705352783, -0.1263846457004547, -0.6323890089988708, 1.0625015497207642, 0.16566503047943115, 0.7728447318077087, -1.0928211212158203, 2.4126813411712646, 4.396603584289551, -4.8967509269714355, 2.626265048980713, 0.513595461845398, -0.7583998441696167, -0.46890872716903687, 1.4837168455123901, 2.4646048545837402, 1.422412633895874, 2.4902775287628174, -2.852484703063965, 2.710486888885498, -0.9648925065994263, 0.6621097922325134, -2.2732675075531006, 0.6535074710845947, -0.5179876089096069, -1.3421366214752197, -3.01302433013916, 1.0682282447814941, -3.0621814727783203, 1.3302984237670898, 0.08926387131214142, -2.011770486831665, -0.036141056567430496, 0.9613938331604004, 1.899095892906189, 2.2921102046966553, 1.4849117994308472, 0.9386070966720581, 2.061631441116333, 2.7352333068847656, -1.8511520624160767, 0.2883506715297699, -0.08517581224441528, 1.3292478322982788, 3.3525359630584717, -0.249465212225914, -0.6141724586486816, -1.3812823295593262, -1.7808269262313843, 1.5596978664398193, 1.9724808931350708, 0.20963218808174133, 0.1861705183982849, -2.5596415996551514, -1.2173093557357788, 3.111337900161743, -1.4895786046981812, 0.9543232917785645, 1.3442379236221313, 1.3893158435821533, -0.7943313121795654, 2.305406332015991, 11.413041114807129, -0.2292211800813675, -0.21522825956344604, 0.34487274289131165, 2.844245195388794, -0.4748763144016266, 0.5314624309539795, -4.197497844696045, -1.3820794820785522, -3.1503376960754395, -2.0905845165252686, 1.1512924432754517, 7.706011772155762, -2.9003093242645264, 0.04839005321264267, 1.6661856174468994, 2.130556106567383, 2.6237869262695312, 0.1288883090019226, -2.8089332580566406, 0.7492856383323669, -4.897220134735107, 0.8055568337440491, -2.1137638092041016, 1.5437476634979248, -1.1828628778457642, -0.7636539340019226, -1.0235302448272705, -0.5911881327629089, -2.43742299079895, 6.381912708282471, -0.48916059732437134, -0.7968547344207764, -1.8754898309707642, 1.3829759359359741, -4.785099029541016, -1.8151752948760986, 5.504164695739746, -0.633293867111206, 1.2593693733215332, -1.0744322538375854, -1.6371523141860962, 0.19211214780807495, 2.0595014095306396, -2.4880409240722656, -1.1252573728561401, 1.186768889427185, -4.250030994415283, -4.332525730133057, -1.1191437244415283, 0.026469310745596886, 0.09911532700061798, -2.129366874694824, -1.273409128189087, 0.8979065418243408, 1.146240234375, 0.7821990251541138, -1.211782693862915, 2.4033777713775635, -3.0687952041625977, 1.5147483348846436, -1.19257652759552, 0.41678741574287415, -2.5681660175323486, -3.106332302093506, 0.49681729078292847, 0.29707396030426025, 2.054490804672241, -2.1717469692230225, -3.2881503105163574, -1.778974175453186, -1.6731038093566895, 2.530271053314209, -4.743654727935791, 1.6483168601989746, 2.7251322269439697, 1.1762465238571167, -1.0258841514587402, -0.2004707008600235, 1.0900142192840576, 6.69807767868042, 3.0760395526885986, -1.9629944562911987, 0.12248002737760544, 2.1999950408935547, -8.010636329650879, 2.2923424243927, 1.9325765371322632, 0.22245804965496063, -0.0704745277762413, 0.8980762362480164, -1.6683754920959473, -1.22008216381073, 0.9876181483268738, -1.546162724494934, 4.7301506996154785, -2.0758917331695557, 9.306678771972656, -2.2947566509246826, -0.6160925626754761, -0.09843488782644272, 1.7377262115478516, -1.0816174745559692, 0.8182050585746765, -0.6867411136627197, -0.49895671010017395, 3.7509942054748535, -0.6083381772041321, 1.4871271848678589, -1.4925267696380615, -1.2504128217697144, -0.8112029433250427, 1.1137875318527222, -1.6138567924499512, 1.0009266138076782, -1.086132287979126, -1.055661678314209, -0.6035140156745911, -2.5974087715148926, -2.5302319526672363, -1.8080202341079712, -3.954864978790283, 0.3856636881828308, -0.6442080736160278, 0.7510145902633667, -0.9665552377700806, 0.4319287836551666, -3.7848241329193115, 0.43536561727523804, -2.7907397747039795, -0.8359206914901733, 1.5354503393173218, 4.163453578948975, 1.9692411422729492, 0.30295678973197937, 3.1358349323272705, 1.231092929840088, 2.3212695121765137, -0.24198327958583832, -0.8610200881958008, -7.519371509552002, 9.728129386901855, -1.254696249961853, -0.2787427008152008, 1.9395679235458374, -2.842975616455078, -0.7524898052215576, 1.1329032182693481, 0.010285712778568268, -0.004720444791018963, 0.8719428181648254, -1.547980546951294, 2.3705968856811523, 2.712550163269043, -2.310699939727783, 1.146446943283081, -0.3399665951728821, 2.6284053325653076, 1.5930670499801636, -2.8549983501434326, -1.5221312046051025, -4.088089942932129, -7.07492733001709, 0.4198545515537262, -2.5478405952453613, 1.1950089931488037, 1.7297618389129639, 0.2882082462310791, 0.5889989733695984, -0.4509068727493286, 2.468003273010254, 3.5178353786468506, 2.1576220989227295, 1.6025100946426392, 1.3712451457977295, -0.6905012130737305, 1.0740684270858765, 0.09417645633220673, 2.568881034851074, -1.4905472993850708, -2.0770671367645264, 5.104503631591797, 2.672672986984253, -0.9236987233161926, -1.5155179500579834, 0.04143522307276726, -2.3413937091827393, -0.7555857300758362, -1.8378815650939941, 1.8763800859451294, 1.8670185804367065, 7.6804938316345215, -0.25659486651420593, 1.94672429561615, -2.2255561351776123, 2.532294988632202, -2.1216020584106445, 3.2686166763305664, -0.36557912826538086, -1.8645097017288208, 0.29111915826797485, -5.678169250488281, -0.8030181527137756, 0.2296804040670395, 2.7196238040924072, 0.2156086415052414, -1.8408280611038208, 6.146528720855713, 1.6206274032592773, 1.3798837661743164, -0.9991478323936462, 0.48007917404174805, 4.054282188415527, -6.266354560852051, 0.9604368209838867, 1.7128738164901733, -4.231009483337402, 0.9519550204277039, 0.8977174758911133, 0.5939522981643677, -0.2385418713092804, -0.2342560887336731, 0.39250215888023376, 0.2701541483402252, -0.2894200384616852, 4.501354217529297, -5.135410785675049, -5.072408199310303, 2.3066155910491943, -0.7652429938316345, -3.230145215988159, -3.876788854598999, -3.082451820373535, 4.464818000793457, 1.4537599086761475, 0.7293774485588074, -1.6904038190841675, 0.5031287670135498, -1.7600934505462646, 5.371610164642334, 2.5081982612609863, -0.5853852033615112, 0.721890389919281, 3.7366015911102295, 0.15651835501194, -1.4511809349060059, 0.3850747048854828, 3.9949135780334473, 1.104596495628357, 4.453344821929932, -3.2542927265167236, -0.6729962825775146, -0.6007411479949951, 3.188778877258301, 0.030883684754371643, -0.137869194149971, 1.6399558782577515, 2.9840383529663086, 0.33556661009788513, 0.34647125005722046, 1.6706995964050293, 3.18208646774292, 1.2674061059951782, 1.4359304904937744, -2.449363946914673, 0.19237184524536133, 0.38261663913726807, 0.011858767829835415, -1.0535078048706055, -0.03127627074718475, 4.409191131591797, -3.1071441173553467, 0.610789954662323, 2.718621253967285, -1.059362769126892, 2.0155928134918213, 1.4113900661468506, 3.830732822418213, 1.6212769746780396, -0.46562880277633667, -1.171388864517212, 0.45549291372299194, -0.8043472170829773, -3.371253490447998, -2.789414405822754, -1.3055955171585083, -2.4774105548858643, 2.4442076683044434, 0.22042840719223022, -0.6448431015014648, -0.3708983063697815, -3.3701744079589844, 2.4624814987182617, 1.3675546646118164, 0.3908068537712097, -8.83596420288086, -0.5381466746330261, -1.1903154850006104, 1.0728545188903809, 0.6793572902679443, 2.0214765071868896, 1.6429706811904907, 1.1023099422454834, 2.2745394706726074, -3.1281421184539795, 1.4187566041946411, 0.5822104215621948, -0.23720119893550873, 5.7656097412109375, -2.8914644718170166, 0.16760170459747314, -2.337033987045288, -1.868865966796875, -4.261861324310303, 3.9333314895629883, -2.8477678298950195, -3.3692824840545654, 2.0367627143859863, 3.008331775665283, -3.4661593437194824, 2.3076164722442627, 2.624248743057251, 0.5586022734642029, -2.4016506671905518, -2.1673080921173096, 1.2370887994766235, -0.6368021368980408, 3.102111577987671, -1.3828378915786743, 0.7529205083847046, 3.092914342880249, -1.6941287517547607, 4.426268577575684, 1.3234405517578125, 0.9561704993247986, -7.585238456726074, -4.569916248321533, 1.6048632860183716, 2.016327142715454, 2.3881962299346924, 1.890381097793579, -0.2529413104057312, 3.4354827404022217, 1.3688414096832275, -2.9341647624969482, -1.7012635469436646, -0.8731071949005127, -6.638580799102783, 1.7647123336791992, 3.292329788208008, 1.347612977027893, 1.294965147972107, -0.8086761832237244, 10.751537322998047, 0.4737236201763153, -0.7665107846260071, 3.83025860786438, 1.4859472513198853, -1.8327091932296753, -0.14558960497379303, 0.7081750631332397, -2.1381890773773193, -0.6324376463890076, 3.6675772666931152, 2.8530571460723877, -0.44248586893081665, 0.3295838236808777, -4.566123962402344, 3.605668306350708, -7.719541072845459, 0.30022260546684265, -2.082197904586792, 1.5649609565734863, -7.134665489196777, -0.03985443711280823, 2.0867598056793213, 0.6427156925201416, 6.209408760070801, -0.013405868783593178, -2.869262218475342, 0.7840962409973145, -0.42070791125297546, 6.403656959533691, -0.49672701954841614, 0.8969538807868958, 0.9298424124717712, 4.042385578155518, 2.034600257873535, 2.030799388885498, 0.2457214891910553, 1.7701565027236938, -1.2078276872634888, 1.5175142288208008, 3.4242026805877686, -0.14760135114192963, -2.621185779571533, 0.5467403531074524, 0.9047471284866333, -0.6127266883850098, 0.8919049501419067, -1.2207351922988892, 1.5173736810684204, -4.110660552978516, 3.5126659870147705, 5.0323638916015625, -0.43090251088142395, 0.5405949354171753, 0.4900456666946411, 3.976741075515747, -6.940736770629883, 0.6512630581855774, 1.5512669086456299, -0.7521755695343018, 0.032703422009944916, -0.8253582119941711, 5.995931625366211, 1.935325264930725, 1.885806679725647, -4.353226661682129, -2.931774139404297, 0.14414863288402557, -0.7062562704086304, 0.39541512727737427, 4.714181900024414, -0.463683545589447, -1.17667818069458, 0.20132559537887573, -5.895747661590576, 4.069644927978516, -2.9208037853240967, -3.8181567192077637, -0.20271337032318115, 0.6640790700912476, -1.3188081979751587, 2.598609685897827, -1.8729676008224487, 0.6823604702949524, -2.01454496383667, -0.6586058139801025, -0.5819252729415894, 3.3901472091674805, 4.014570713043213, -2.7026915550231934, 1.2579439878463745, -1.336837649345398, -2.7066922187805176, -2.059439182281494, 0.10771875083446503, -4.144497871398926, 4.045501708984375, -2.180515766143799, 4.703638553619385, -1.868435025215149, 0.5305465459823608, 1.9174182415008545, 0.9711428284645081, -6.275602340698242, -1.4803078174591064, 0.5572511553764343, -1.332801342010498, 0.24135002493858337, 4.251543998718262, 1.606101155281067, 2.9995970726013184, -3.542898178100586, 1.2291494607925415, -0.4189090132713318, -4.02554988861084, 0.07881321012973785, 0.3644857406616211, -2.0767135620117188, 0.3626600205898285, -0.5029283165931702, 1.6976004838943481, 8.459667205810547, -2.128636598587036, 1.6751136779785156, 0.9697030186653137, -3.8855795860290527, -1.4667397737503052, 3.578958749771118, -3.714104175567627, 0.627244770526886, -2.9879279136657715, 1.4364265203475952, -2.840041160583496, -3.3125674724578857, 3.485830307006836, -0.3732694983482361, -4.997372627258301, 0.525530993938446, -4.330155849456787, -4.908960342407227, -2.9694745540618896, 5.880131244659424, 2.3366010189056396, -1.2269219160079956, 2.4656546115875244, 0.17323707044124603, -0.2006356567144394, 1.6391209363937378, -2.107404947280884, 4.834728240966797, 0.10775958001613617, 0.28046831488609314, -2.3143832683563232, -0.05738097056746483, 1.3665013313293457, -0.8339259028434753, -1.3176499605178833, -0.2911984920501709, 1.189853549003601, 0.5413323640823364, -1.0790239572525024, -1.7152037620544434, -1.7359782457351685, -1.4167029857635498, -0.2406318187713623, 1.076621413230896, 2.3381805419921875, 0.4707939922809601, 4.89475154876709, 0.6005595326423645, -4.455596446990967, -0.2669624090194702, 2.2844078540802, -3.830409288406372, -0.9381898045539856, -1.3086603879928589, 2.763932704925537, 7.574263572692871, -1.8372303247451782, 0.6717897057533264, 3.491345167160034, 2.685978889465332, -2.561101198196411, -0.7549795508384705, -2.1218268871307373, 0.84244704246521, 2.3251380920410156, -0.21908743679523468, 1.833602786064148, -2.551767587661743, -0.06349565088748932, -5.3088788986206055, 0.2884262204170227, -2.9978702068328857, -3.0990469455718994, -0.10295429825782776, -0.7562758922576904, -0.1479339301586151, 4.371537208557129, 3.4245593547821045, -3.0172371864318848, 1.7758784294128418, -1.4143872261047363, -5.73197603225708, -5.8437604904174805, -0.9192591309547424, -0.43755361437797546, 2.4117565155029297, 4.801471710205078, 0.5770664215087891, -3.6616573333740234, 0.7773327231407166, 2.7605817317962646, 2.309648275375366, -2.353755474090576, 5.79766845703125, -4.139508247375488, 2.992431879043579, 4.001681804656982, 4.469965934753418, -0.14401768147945404, -1.7346853017807007, -1.4728095531463623, -3.858271360397339, 0.8972586393356323, -0.612378716468811, 1.88779878616333, 2.0201754570007324, -1.5467305183410645, -2.3268353939056396, -5.897401332855225, -2.081951141357422, -4.805081367492676, 3.385493755340576, -0.5117175579071045, 0.9114294052124023, 1.522219181060791, 1.7666712999343872, 0.816305935382843, -0.7010661959648132, 1.5857945680618286, 1.0638099908828735, 0.710168182849884, -0.8283538222312927, -2.996431350708008, 1.143913984298706, 2.3978865146636963, 5.1228251457214355, 1.68354332447052, 4.710037708282471, -5.704023361206055, -2.5813026428222656, -7.702846050262451, 3.7609286308288574, 0.655594527721405, -0.5068759322166443, 1.5394983291625977, 1.6343090534210205, -3.246742010116577, 1.827597975730896, -2.486239194869995, 2.3889386653900146, 3.1488842964172363, 0.17460547387599945, 0.5511889457702637, -2.8796632289886475, 3.6403088569641113, 3.090481996536255, -1.8342182636260986, -1.661659836769104, -2.0278303623199463, 3.8492767810821533, -0.049636032432317734, -6.107757568359375, -0.7817444205284119, -0.726319432258606, -0.5047093033790588, 0.8541361689567566, -1.965282678604126, -0.05493729189038277, -2.296539068222046, 5.970638275146484, 0.9101037383079529, -0.1115402951836586, -1.1997013092041016, 3.789560079574585, -0.2120104730129242, -1.494209885597229, -1.829843521118164, 0.09337888658046722, 0.6689540147781372, -0.6275238990783691, -1.4005160331726074, 3.324679136276245, -0.08861690759658813, 1.5793293714523315, -4.374172210693359, -1.4592292308807373, 2.243558645248413, -3.2864155769348145, 0.41063451766967773, -1.5273549556732178, 0.3847494125366211, -0.0008032183395698667, 0.5579624772071838, -0.7161447405815125, -0.5391759872436523, -1.2470662593841553, -1.2420979738235474, 1.315377116203308, -1.8294885158538818, 0.0979522094130516, -0.7920389771461487, -1.1886519193649292, 2.1215319633483887, 4.782252311706543, -2.0292015075683594, 5.7686848640441895, -2.1180362701416016, -0.6150714755058289, 3.259042263031006, 0.3615000545978546, -0.8544715642929077, -0.8855330944061279, 2.7627201080322266, 2.793306827545166, -0.1399075835943222, 0.7664670944213867, -0.7241490483283997, -1.540656566619873, -4.434791088104248, -2.3615691661834717, -0.6575142741203308, -0.14900583028793335, 4.082336902618408, 3.106558084487915, -2.9595134258270264, -1.9838786125183105, 0.5779916048049927, 3.708519458770752, -0.334204763174057, -0.8089558482170105, -4.231070518493652, -1.5401972532272339, 1.8442445993423462, -2.119413375854492, 2.7687265872955322, -0.49279072880744934, 1.004412293434143, 0.9716989398002625, 0.2853028178215027, -3.1562769412994385, 1.4144593477249146, -0.797335147857666, -0.5973351001739502, -2.293025255203247, -0.21969352662563324, 1.6924818754196167, 0.39605212211608887, 1.6171025037765503, -1.1716259717941284, -0.39657261967658997, -1.944156289100647, 10.237591743469238, 1.3607008457183838, 1.5573915243148804, 0.5681761503219604, -0.024186111986637115, -1.009007453918457, 0.8425731658935547, -2.2131590843200684, -1.4780298471450806, -1.5938178300857544, -1.8800666332244873, 0.2078031450510025, 0.44316092133522034, -4.169859886169434, 3.1801598072052, 0.058951329439878464, -1.5799916982650757, 1.728417992591858, -3.5226328372955322, 1.1094805002212524, -5.217529773712158, -2.5264055728912354, 2.5701639652252197, 3.0952858924865723, -2.08727765083313, -4.123987674713135, 1.5708177089691162, -4.441165447235107, -2.0170934200286865, -0.3008106052875519, -0.0024497411213815212, 3.4303834438323975, -3.4777321815490723, -2.549203395843506, 0.47157204151153564, 0.211033433675766, -4.266839027404785, 4.574240207672119, 1.5972200632095337, -0.9756978750228882, -2.351912021636963, -0.7769690752029419, 3.9922523498535156, -6.843135833740234, 4.471722602844238, -1.0084441900253296, 0.7979512214660645, 0.7842139601707458, -1.2787933349609375, 2.5388643741607666, -3.6241612434387207, 1.103256106376648, 4.474771976470947, -1.5625234842300415, -1.123605489730835, -1.75730299949646, 0.6239703297615051, -2.062227487564087, 0.7838470339775085, -0.045351773500442505, -2.0698323249816895, -4.294053077697754, -2.305311918258667, -0.6829829812049866, -0.8157951831817627, -0.5335088968276978, -1.3503329753875732, -2.3407249450683594, -0.636052668094635, 0.6916106939315796, -2.408602714538574, -1.4061863422393799, -3.100977897644043, -1.3482755422592163, -2.392714738845825, -1.6367989778518677, 1.1632001399993896, -0.7313956618309021, -1.6731305122375488, -0.16672316193580627, 0.3647259771823883, 0.3710935115814209, 2.2590491771698, 4.654627799987793, 0.7959023118019104, 2.9818050861358643, 2.7948055267333984, -4.419848918914795, 1.9236313104629517, 0.2446174919605255, 2.6018800735473633, -5.823288440704346, 1.6753785610198975, 6.15243673324585, 5.280145645141602, 2.147509813308716, 1.8286373615264893, 0.25637343525886536, 1.4171855449676514, 1.1066935062408447, 0.44238635897636414, 4.137646675109863, -0.24901573359966278, 2.616584539413452, -1.2676011323928833, 0.6969446539878845, -0.9959392547607422, 0.98552405834198, -2.27537202835083, -0.1791011542081833, -2.665724754333496, -2.4464001655578613, -4.307199954986572, 2.2951624393463135, -1.5650526285171509, 3.4570016860961914, 0.8538132309913635, 4.172774314880371, 1.3013614416122437, -1.547412633895874, -0.6631999611854553, -0.019498175010085106, -0.535698652267456, 2.6502130031585693, -1.4177026748657227, -2.928288221359253, -0.9481407403945923, 1.6234561204910278, 3.64414119720459, -2.611706018447876, 2.8643782138824463, -1.9828252792358398, -3.3817946910858154, 0.12757009267807007, 4.897281169891357, 3.8631110191345215, -2.545074701309204, -0.5864357948303223, -1.4141207933425903, -0.22275324165821075, -0.16530364751815796, -2.4274842739105225, 0.39923909306526184, 3.173957109451294, -3.5985331535339355, -0.7088702321052551, -3.2244083881378174, -2.662647008895874, 1.947019100189209, 2.3898401260375977, 1.4671845436096191, 3.374751091003418, -0.8736686110496521, -0.18125639855861664, 9.604293823242188, 7.04922342300415, 2.8969223499298096, 3.1249277591705322, -1.6971708536148071, 3.0657145977020264, 1.9062331914901733, -3.087491273880005, -1.585015058517456, 0.3622317910194397, -3.214737892150879, 0.7647994160652161, 0.20972974598407745, -0.5727099776268005, -0.09682830423116684, 2.081156015396118, 1.93827223777771, 0.06696378439664841, 2.192955255508423, -1.5529465675354004, 0.85103839635849, -0.9976760745048523, -0.05120813101530075, 3.1424715518951416, -2.0255300998687744, 1.6228411197662354, 0.6530956625938416, 0.7825336456298828, 1.1021445989608765, -2.149001121520996, 2.324144124984741, -3.156205892562866, 2.713954210281372, -1.3585115671157837, 0.6130371689796448, 2.5882091522216797, 0.20752738416194916, 0.7513731718063354, 1.3375742435455322, -3.371846914291382, 0.8111258745193481, 1.0679821968078613, 1.9680070877075195, -0.8380261659622192, -5.623980522155762, -0.9284883141517639, 3.972414493560791, -0.5870640873908997, 0.909260094165802, -0.09706566482782364, -0.5096177458763123, 0.12504130601882935, -3.2812511920928955, -0.7551570534706116, 5.00181770324707, 0.1422749012708664, 4.334377288818359, -1.723982572555542, 0.7618005871772766, -0.39879459142684937, 3.058725118637085, 2.0591413974761963, 2.480191946029663, -0.1266961544752121, 2.8131120204925537, -4.514163494110107, 0.5459444522857666, 3.2027790546417236, -1.0226917266845703, -3.2274537086486816, 0.929853618144989, -3.313041925430298, -0.3804596960544586, 0.8899382948875427, -1.034106969833374, 0.32354119420051575, 4.655855655670166, 1.4414502382278442, -10.236523628234863, 2.3661067485809326, 0.5304171442985535, 2.6635334491729736, -2.0620288848876953, 0.9977558851242065, 1.6773148775100708, -1.33624267578125, 3.9102814197540283, 3.968351125717163, -1.7648873329162598, -1.3859003782272339, -0.918423593044281, -0.46104681491851807, 1.430416464805603, -0.6218293905258179, 1.484818935394287, -4.995750904083252, -3.121488571166992, -0.9374595880508423, 4.602000713348389, 2.498243570327759, 2.581108331680298, -1.8723928928375244, 2.15775728225708, 3.119229316711426, -2.068406343460083, -2.1022613048553467, 3.2747113704681396, -1.1554467678070068, -2.3827102184295654, -6.529304504394531, 1.0531624555587769, -0.7079370617866516, 2.8829479217529297, -1.7669575214385986, -4.908668518066406, -0.2832948863506317, -4.090022563934326, -0.2965739965438843, -3.687605381011963, -0.7425974011421204, 2.9810080528259277, 2.5801026821136475, -0.7147046327590942, 3.87628436088562, 0.8532204031944275, -1.9368914365768433, -3.164114236831665, 1.9846432209014893, 1.391271948814392, 0.8376381397247314, 3.5554521083831787, -4.120561122894287, -3.9349470138549805, 0.4055354595184326, -3.430128574371338, -2.8129212856292725, 4.0429768562316895, -0.12995727360248566, 0.40867871046066284, -2.378337860107422, -2.601270914077759, -2.627285957336426, -1.3185153007507324, 0.02138078585267067, 3.461271047592163, 0.4841960668563843, 0.37173813581466675, -3.1811764240264893, -3.6860454082489014, 5.7553791999816895, 1.7361899614334106, 0.9298120141029358, -3.627776861190796, -0.8576356768608093, 3.355086088180542, 0.7846934795379639, 1.6960734128952026, 1.30965256690979, 2.4307589530944824, 1.0298157930374146, -4.650915145874023, -5.46401309967041, 2.4398863315582275, -0.6153796911239624, -0.876998245716095, -6.0396728515625, -3.1286284923553467, 2.064129114151001, 3.233612298965454, 0.859213650226593, 0.42076733708381653, -0.15137861669063568, 2.3075344562530518, -2.0764455795288086, 0.8599209189414978, 0.7032397985458374, -0.5166532397270203, -2.12139630317688, 2.7075626850128174, -6.449855804443359, 0.30642426013946533, -1.4174206256866455, -2.2336246967315674, 4.155194282531738, -1.316235065460205, -0.710819661617279, -0.1181577667593956, -1.3191554546356201, 0.43706777691841125, -5.097136497497559, -6.2675676345825195, 6.5139360427856445, -2.552971124649048, -0.42686328291893005, -2.081897735595703, 1.5103583335876465, -0.2828449606895447, -4.439703464508057, 0.2026389241218567, 0.34246358275413513, -2.3041179180145264, 0.5953144431114197, -0.24669155478477478, -0.6991668343544006, 2.1250319480895996, -0.6019701361656189, 5.713282108306885, 0.8027239441871643, -0.8525868654251099, 1.3751269578933716, 2.6781680583953857, -1.6014997959136963, -0.9085575342178345, -1.4505165815353394, -1.6265921592712402, 2.5366156101226807, 6.091746807098389, -2.677569627761841, 1.6646392345428467, -1.1775493621826172, -4.958043098449707, -3.7044906616210938, 4.49333381652832, 2.306682586669922, 1.3044322729110718, 1.010689377784729, -0.20225462317466736, 0.9825312495231628, -0.8436576724052429, 1.3814777135849, 0.4798614978790283, 0.21009784936904907, -1.8625997304916382, 2.106337785720825, -1.6756370067596436, 0.6357982754707336, 0.6668422222137451, 2.9220564365386963, -0.7430495023727417, 2.819916009902954, 0.1434338539838791, 2.268254518508911, 0.9532503485679626, -1.6515334844589233, 3.4189116954803467, 1.7843624353408813, -0.1333933025598526, 0.9865400791168213, -2.369469165802002, -5.198314666748047, -0.35534220933914185, -1.1571606397628784, 7.075150966644287, 2.0053460597991943, -6.478203773498535, -3.9176337718963623, -0.3422278165817261, 1.0777368545532227, -1.0806444883346558, -2.04238224029541, -3.304755687713623, -1.1798816919326782, -0.9945899844169617, 5.887284278869629, 2.9168202877044678, -0.13933981955051422, 1.8187055587768555, -2.211960792541504, 3.54679274559021, -3.1302311420440674, -1.644795536994934, 1.7798317670822144, 5.06986141204834, 2.795342206954956, -1.289596438407898, -0.29334425926208496, -1.236824870109558, -2.5126237869262695, 0.9172226786613464, 0.3670457601547241, 3.0029284954071045, -0.6467733979225159, -1.7036973237991333, 2.6022331714630127, -0.6501913070678711, -1.4954235553741455, -2.050300359725952, 2.9810729026794434, 0.2592601776123047, 0.7338340878486633, 2.046231269836426, 0.8247287273406982, 3.640094518661499, -2.1840922832489014, -1.4785517454147339, -0.6133412718772888, 2.484527587890625, 1.866465449333191, 0.7858410477638245, -0.809096097946167, -1.4726859331130981, 1.289655327796936, -4.463536739349365, 0.4059029817581177], "xaxis": "x", "y": [0.2619815766811371, -1.6471887826919556, -2.1036012172698975, -0.6661433577537537, -0.6832173466682434, -1.740047574043274, -1.2150944471359253, -1.2653809785842896, -0.5759990811347961, -3.177859306335449, 0.7420580983161926, -0.9897416234016418, -1.4218940734863281, -0.2799414396286011, -3.997526168823242, -0.4451850354671478, -0.5240417122840881, 0.4399271607398987, -1.1932579278945923, -0.06827422231435776, -1.345824956893921, -1.4195048809051514, -1.2290962934494019, 0.15714773535728455, -1.350852608680725, 0.06973151117563248, -0.41902756690979004, -0.7833974957466125, -0.9379794597625732, -1.8995174169540405, -1.251509428024292, -1.3173142671585083, -1.6201145648956299, -1.0664454698562622, -0.6624378561973572, -1.9403983354568481, 0.44911473989486694, 0.8219304084777832, -3.4512224197387695, -1.0134121179580688, -0.9393342733383179, -1.693152904510498, -2.1143910884857178, -1.6934361457824707, 3.169785261154175, -1.4500913619995117, -0.9736624360084534, -2.332634210586548, -1.1855183839797974, 0.6450775265693665, -2.0115573406219482, -0.5489679574966431, -1.814895510673523, 0.2903369069099426, -0.6366676688194275, 1.08389413356781, -0.013613074086606503, 1.0195376873016357, -0.16784422099590302, 0.38186389207839966, -1.8374806642532349, -1.174332618713379, 1.0951482057571411, -1.166878581047058, 0.5911089181900024, -0.27201196551322937, -0.5430015921592712, -1.0734267234802246, -2.49811053276062, -3.7225005626678467, 0.25452566146850586, -0.764388382434845, -0.569571316242218, -2.0150697231292725, 0.7867501378059387, -1.3961085081100464, -2.8288469314575195, -0.5900258421897888, 0.7373807430267334, -1.1236201524734497, 0.34262606501579285, 0.1392112523317337, -1.725979208946228, -0.7446568608283997, -1.4693892002105713, -1.9317244291305542, 0.010689729824662209, 0.3048088848590851, -0.249357670545578, -1.0141825675964355, -2.0677099227905273, 0.07409138232469559, -2.5245308876037598, -0.2136724591255188, -0.3669297695159912, -1.6533178091049194, -1.1763184070587158, -0.4863852262496948, -0.42117181420326233, -0.053438857197761536, -1.0728939771652222, -0.6943196654319763, -0.21165108680725098, 0.13574643433094025, -0.5754602551460266, 0.5770111083984375, -1.1315946578979492, -0.9249434471130371, -0.5885160565376282, -0.6889265179634094, -1.1146751642227173, -1.0498579740524292, -0.7891272306442261, -0.5749197006225586, -0.8975950479507446, -1.1640127897262573, -1.899795413017273, -0.7335710525512695, -0.2859787046909332, -1.2467161417007446, 0.5284103751182556, -0.16740430891513824, 1.6699632406234741, 1.0378098487854004, -0.9293215274810791, -0.7448735237121582, -1.0768601894378662, 1.3982142210006714, -0.7647204995155334, -0.022887596860527992, -0.7452913522720337, -0.9966577291488647, 0.09255146235227585, -0.7752229571342468, -0.7662309408187866, -0.5606057643890381, -1.2954474687576294, -0.7218523621559143, -1.3141019344329834, -0.32696425914764404, -0.7205984592437744, -0.41175922751426697, -0.27818095684051514, -0.8423675298690796, -0.5293424725532532, -0.7447137832641602, -0.2138502448797226, 1.090835452079773, 0.07677426189184189, 0.25510942935943604, -0.9255411028862, -1.8944981098175049, -0.498070627450943, -0.9964573979377747, 0.8321912288665771, -0.7775372266769409, -2.8222968578338623, -0.31611257791519165, -0.22034816443920135, 0.5636508464813232, -0.13492536544799805, 1.0844122171401978, -0.3649807870388031, -0.6979559659957886, -1.4840203523635864, -0.40025395154953003, -1.1991214752197266, -0.9816694259643555, -0.059593237936496735, -2.0875942707061768, 3.2172558307647705, -1.2235881090164185, -0.32418856024742126, 0.35014083981513977, -1.0935355424880981, -1.0742266178131104, -1.4794402122497559, -0.5042915344238281, 0.09373117238283157, -3.6439976692199707, 0.8462085127830505, 0.14850389957427979, -1.3899240493774414, -0.5136978030204773, 0.7510098218917847, -0.7073996067047119, -2.19248104095459, 1.3714557886123657, -0.6138195991516113, -0.1476794332265854, -0.2230350375175476, -0.582118809223175, -1.4759927988052368, -0.26874417066574097, 5.227797508239746, 0.03015068545937538, -2.0849175453186035, -2.438617706298828, 0.02199247106909752, 0.792873203754425, -1.514081358909607, -0.6094028949737549, -2.758026361465454, -0.6320632696151733, -0.16136229038238525, 0.7526673674583435, -0.8418930172920227, -0.14845560491085052, -0.373333603143692, -0.08377698808908463, -0.3710382580757141, 0.9959138035774231, -0.3747006356716156, -2.957289695739746, -0.6613986492156982, 0.027972165495157242, -0.22739055752754211, -0.2019215226173401, -0.3618573248386383, -1.4942636489868164, -0.12698602676391602, -1.4208893775939941, 0.8344035744667053, -0.09624438732862473, 0.27590057253837585, 0.20482513308525085, -0.46845853328704834, -0.7153528332710266, 0.793253481388092, -1.4546887874603271, -0.9396114349365234, -1.8149693012237549, -0.3870469629764557, -0.5405373573303223, 1.8788549900054932, 0.9508459568023682, -1.952118158340454, -0.40335240960121155, -0.2227088063955307, 0.08441773056983948, 0.044594213366508484, -0.6687074303627014, -1.6755000352859497, 0.9995788335800171, -0.5702126622200012, -1.0072754621505737, 0.17875123023986816, 1.0843796730041504, 0.6793592572212219, -0.6312662959098816, -1.28281831741333, 0.6409767270088196, 0.06742434203624725, -1.2286738157272339, 0.10666415095329285, 0.5670335292816162, 0.3443731963634491, -1.1975237131118774, 1.3142021894454956, 0.8469449281692505, 0.6247583031654358, -1.5434311628341675, 0.17754915356636047, 0.3790983259677887, -1.0706223249435425, -0.09230925142765045, -1.7327039241790771, 1.232891321182251, -1.010109543800354, -0.3656955361366272, -0.7300193905830383, -0.09776309132575989, 0.9946843385696411, -0.6264845132827759, -0.6524887681007385, -0.8242847323417664, 0.5602701902389526, -0.8730382919311523, -0.6248190402984619, 0.45960232615470886, -0.22824621200561523, -1.3278356790542603, -1.8388047218322754, -0.22589224576950073, 1.1494096517562866, -0.08899851888418198, -0.13225170969963074, -0.4519738554954529, -0.9276074767112732, 0.6692712306976318, 1.0509358644485474, -0.19184479117393494, 1.3374205827713013, 0.5622326135635376, -0.4946787655353546, -0.008340952917933464, -0.4461495876312256, -0.7412900924682617, -0.44072335958480835, 1.1944622993469238, -1.0575428009033203, 1.6642802953720093, 1.0193822383880615, -0.3987407386302948, -0.136996328830719, 1.062949538230896, 1.032823920249939, -0.5841451287269592, 0.9280335903167725, -0.8602551817893982, -0.16989898681640625, -0.7798960208892822, 1.1000239849090576, -0.382733553647995, 0.32929500937461853, -0.7274447083473206, -0.9681166410446167, -1.04873526096344, 0.8396363854408264, 0.5108439326286316, -3.5868124961853027, 0.47430428862571716, 0.060600049793720245, -1.2139122486114502, -0.2571795582771301, 0.624030351638794, -2.617760181427002, -0.9669510722160339, -0.4386381506919861, -0.5017699599266052, -1.2615710496902466, 1.9793323278427124, -2.922149419784546, -0.1629301905632019, 0.239195317029953, 0.2989644408226013, 0.05241100862622261, -0.02945077233016491, -0.07548536360263824, -0.6637368202209473, -0.45666319131851196, 2.5747568607330322, 2.104661226272583, 0.587959885597229, 0.3322393298149109, 0.6226934790611267, -0.8625209331512451, -0.595363974571228, -0.20707689225673676, -0.4383501410484314, -1.072082281112671, -1.1539853811264038, -1.0474258661270142, 0.129859060049057, 0.7720456123352051, -0.09344977885484695, -0.006012908648699522, -0.21311527490615845, 0.7592455744743347, -1.6424977779388428, 1.2711052894592285, 0.5653663277626038, -0.5248253345489502, 0.35917696356773376, 0.998764157295227, 2.2291922569274902, 0.38554662466049194, -1.2711676359176636, 0.7292715311050415, -0.32874321937561035, 0.376603901386261, 1.3161876201629639, -1.6232503652572632, -1.1906641721725464, 0.7196987271308899, -0.40254953503608704, 0.9123550653457642, -1.5790611505508423, 0.20750588178634644, -0.19052912294864655, 2.24596905708313, -0.6405874490737915, -0.7006934881210327, 0.582226037979126, 0.7729457020759583, -0.6973321437835693, 0.21808220446109772, 1.164928913116455, 0.036136262118816376, 0.09789365530014038, 0.16599036753177643, 0.14291693270206451, 0.8370308876037598, -1.6721006631851196, -1.5046837329864502, 0.3573634922504425, -0.2842506766319275, -1.8823307752609253, -1.2710286378860474, 0.2998640239238739, -0.023018853738904, -2.468845844268799, -2.3309459686279297, -0.05249115824699402, -1.49319326877594, 0.01657632365822792, -0.05498968064785004, -0.23484455049037933, -0.0820101648569107, 0.014900055713951588, 0.6317074298858643, -0.9763768911361694, -1.4677590131759644, 0.7654825448989868, 0.3286695182323456, -0.3467477560043335, -1.1772631406784058, -0.513056218624115, 0.20748397707939148, -0.8824465870857239, 1.0523258447647095, -0.49461865425109863, -0.44063299894332886, 0.012888109311461449, -0.73898845911026, -0.26792123913764954, 0.5080331563949585, 0.6254165768623352, -0.39317700266838074, -1.8477904796600342, -0.4537384510040283, -0.14985422790050507, 0.8372008204460144, -0.6084901690483093, -1.340260624885559, 0.5798927545547485, 1.431138038635254, 0.07586951553821564, -1.4274795055389404, -0.02645999751985073, 0.556904673576355, -0.9630637764930725, -0.8007047176361084, 2.0155527591705322, -0.4433938264846802, 0.784530520439148, -0.08875418454408646, 2.1344540119171143, 0.9543256163597107, 0.0775386318564415, -0.36609312891960144, 0.3921617567539215, -0.7194134593009949, -0.365307092666626, -0.3265182673931122, -0.8355855345726013, -0.08419560641050339, -0.47415366768836975, -0.758423924446106, 0.20818312466144562, 0.4514424502849579, -0.7819390296936035, 0.352638304233551, 0.009741012938320637, 0.4381897449493408, -0.11080989241600037, 0.36495712399482727, -0.23454269766807556, 1.1014745235443115, 0.20156696438789368, -0.189828559756279, 0.32724589109420776, -0.6120955944061279, 0.026012752205133438, 0.29757237434387207, -0.41720664501190186, -0.8546890616416931, 0.5573194622993469, 0.26785919070243835, -0.9977611303329468, 0.08393742144107819, 0.03616902977228165, 0.7844832539558411, -1.2044116258621216, 0.9017897844314575, -0.7336974143981934, 0.2833588123321533, -0.2687465250492096, -0.5055956840515137, 0.8007528781890869, -0.7018733024597168, 0.11301770061254501, -0.683800458908081, -0.1910858303308487, -0.2849821150302887, -0.5725252628326416, 0.773635745048523, -0.0940103754401207, 2.128094434738159, -0.6264337301254272, 0.10072695463895798, 1.1511081457138062, 0.9031020998954773, 0.4104883074760437, -0.07468397915363312, 0.6579774022102356, -0.15412798523902893, 1.0585391521453857, -1.1291732788085938, -0.16679076850414276, 1.6129260063171387, 0.45291638374328613, -0.561772346496582, 0.8960058093070984, -0.2337885946035385, 0.7517781853675842, 0.3103400468826294, -0.4445074796676636, 0.39545202255249023, 0.6759446859359741, -0.9217973947525024, -0.8448569774627686, 0.629593014717102, -0.04795222356915474, -0.8346806168556213, 0.44338035583496094, 1.5600799322128296, -1.3808519840240479, 0.8814592361450195, -0.9534652233123779, 0.44276437163352966, -0.6262869238853455, -0.845127284526825, 0.5934070348739624, -1.7163959741592407, -0.023371638730168343, 0.024012548848986626, -0.024521708488464355, 0.8398517966270447, -0.8222397565841675, 0.5239773392677307, 0.32531705498695374, -1.039138674736023, 0.4213988482952118, 0.11733561754226685, 0.26277437806129456, 0.38590893149375916, 0.9850348830223083, -0.6385501027107239, -0.9674803614616394, 0.048977065831422806, -0.6188409328460693, 1.0018256902694702, -1.049704670906067, -0.2913084924221039, 1.0286134481430054, 0.026124902069568634, 0.045564185827970505, -1.210483431816101, -0.7900282144546509, 3.6317708492279053, -0.14051540195941925, 0.4711267054080963, -0.5392084717750549, -0.5512691140174866, 2.648078441619873, 1.0769325494766235, 0.020200297236442566, -0.1950562447309494, 0.4424973726272583, -0.21520575881004333, 0.15562281012535095, 0.1388058215379715, -0.7011311650276184, -0.07760484516620636, -0.2789500951766968, 0.8838975429534912, -1.3805177211761475, -0.29627904295921326, -0.9120271801948547, -0.508091926574707, -1.1607797145843506, 1.4356144666671753, -1.9785873889923096, 0.513114869594574, 0.30916810035705566, 0.004343411419540644, 2.7097785472869873, 0.11871205270290375, 0.8009541034698486, -0.06025680899620056, 1.0334250926971436, 1.851730227470398, 0.6329326629638672, -0.17963773012161255, -0.8642716407775879, 0.6354429125785828, -0.8682087659835815, 0.2958466410636902, -0.370540589094162, 0.03441485017538071, -0.08039843291044235, 1.0224555730819702, -0.041459664702415466, -0.006292045582085848, -0.8373897075653076, -1.2898552417755127, -0.308931440114975, -1.3414688110351562, -0.20795537531375885, -0.8811419606208801, -0.8931965827941895, 0.6292821764945984, -0.67679762840271, -0.14364027976989746, 0.8955418467521667, -0.6866409182548523, 2.3654024600982666, -0.9180334210395813, -0.0707603171467781, -0.8374750018119812, -0.9664170145988464, 0.6592922806739807, 0.0980018675327301, 1.1491167545318604, 0.5430245399475098, 0.18391312658786774, 0.39974308013916016, -0.45019325613975525, -0.1741855889558792, -1.1293460130691528, 0.344035804271698, 1.172635793685913, 0.6797958016395569, 2.6828510761260986, -0.9149054288864136, -0.5267794728279114, 0.2895003855228424, -0.4988674819469452, 0.4804961681365967, -0.952122688293457, -1.0917528867721558, -0.7605289220809937, -1.4475749731063843, -0.6809043288230896, -2.4904367923736572, -0.28300943970680237, 0.5874335169792175, -1.1608681678771973, -0.28339219093322754, 0.6047369241714478, -0.7295582294464111, -1.3190672397613525, -0.7958957552909851, 0.535882294178009, 1.0302766561508179, 0.5695008039474487, -0.2493065595626831, 0.6795006990432739, -0.11374933272600174, 0.3487415909767151, 0.49185213446617126, -0.6165757179260254, 0.45371904969215393, -0.5526785254478455, 0.9213181138038635, 2.7388858795166016, -0.5888069272041321, -0.23127862811088562, -0.18587398529052734, 2.120189905166626, -0.10498793423175812, -1.1626293659210205, 0.6846168637275696, 0.26718565821647644, -1.766300082206726, -0.2589915692806244, 0.8961208462715149, 0.2288019061088562, -0.036690033972263336, -0.649290144443512, -1.091537356376648, 0.5953747630119324, 0.04683555290102959, -1.0202507972717285, 0.06162761524319649, 0.7635242938995361, 1.604777216911316, -1.4386857748031616, 0.9491910934448242, -0.7277235388755798, 1.3757259845733643, -0.5916584730148315, 2.431138277053833, 0.3436242640018463, 0.9628992676734924, 0.3104793429374695, 0.2047095000743866, 1.319084644317627, 1.6238281726837158, -0.23346710205078125, 0.3260626196861267, -0.06588646024465561, -0.6879919171333313, -0.5388008952140808, -0.7296947836875916, 0.44266942143440247, 0.7032412886619568, 0.3267529010772705, -0.688095211982727, -0.38229140639305115, 1.0766795873641968, -0.6526832580566406, -0.2709945738315582, -0.1384282112121582, -0.6925219893455505, -0.6692412495613098, -0.5986149311065674, -0.6048027873039246, -0.29239338636398315, -0.27652856707572937, 0.027736511081457138, 0.3424259424209595, -0.01728164777159691, -0.5197791457176208, -1.0565166473388672, 3.258718967437744, 0.7210330963134766, 0.9162926077842712, 0.5691245794296265, 0.3678213655948639, -0.07959175854921341, 0.7667999267578125, 0.5442237854003906, 0.00739695830270648, 1.562990427017212, 0.08264993876218796, 0.5726490020751953, 0.5739207863807678, 0.35563066601753235, -0.018435362726449966, 1.8259209394454956, 0.19465599954128265, -0.8881544470787048, 1.4708092212677002, -0.7629590630531311, -0.5911031365394592, 0.10772975534200668, -2.0164098739624023, -0.6895654201507568, 0.3598935902118683, -1.0473805665969849, -1.1239491701126099, 0.677997887134552, -0.27720415592193604, -0.8949046730995178, -0.21604913473129272, 3.325824737548828, 0.34764984250068665, 0.5863857865333557, 0.1008400171995163, 0.5271801352500916, -0.5852294564247131, 0.4702717661857605, 0.7159873247146606, -0.31316664814949036, 1.3727909326553345, 1.2490861415863037, 0.5130208730697632, 0.6199524998664856, 1.0207568407058716, -0.06718115508556366, -0.300596684217453, -1.4041872024536133, -0.15416491031646729, -0.4688166379928589, -0.11455236375331879, 0.13113455474376678, -0.21690469980239868, 0.4661749601364136, -0.9729048609733582, 0.443940132856369, -0.2602122724056244, 0.33393850922584534, 0.4436633288860321, 0.45387715101242065, 0.1936500072479248, 0.5221895575523376, 0.7191699743270874, -1.355615496635437, -0.5760334134101868, -0.7267686724662781, -0.6293652057647705, 0.34781110286712646, 0.5599018931388855, -0.09739050269126892, 1.6619889736175537, 0.5391550064086914, -0.7569555640220642, -1.770215392112732, 0.12079950422048569, -1.8732455968856812, 1.401231050491333, -0.4748954474925995, 0.05114111304283142, 0.866615891456604, -0.4557923376560211, 0.085580974817276, 0.7005001306533813, 1.6624442338943481, 0.20486292243003845, -0.8034663200378418, 0.3507983684539795, -0.47953107953071594, -0.39616110920906067, 0.12789613008499146, -0.013459625653922558, 0.9186792373657227, -0.038146842271089554, -0.42832818627357483, 0.9404897093772888, 0.422155499458313, -1.4974159002304077, 0.2187158763408661, -0.30005085468292236, 0.2005758136510849, 0.07875522971153259, 0.23116359114646912, -0.48022645711898804, 0.11946681141853333, -0.5071748495101929, -0.6808342933654785, 1.185491681098938, 1.0356204509735107, -0.09224138408899307, 0.2857866585254669, 1.103074550628662, -0.3372778594493866, -0.5273751020431519, -0.08625529706478119, -1.58328378200531, 0.6238875389099121, 1.102332353591919, 0.6842243671417236, -0.681098997592926, 0.6060298085212708, 0.418000191450119, -1.0243204832077026, 0.14050732553005219, 0.3815867006778717, 0.35579755902290344, -0.45611992478370667, -0.07685506343841553, -1.1509590148925781, 0.20411090552806854, -0.02707488276064396, -0.09252438694238663, 0.084666408598423, 0.41958990693092346, -0.573232889175415, 0.05780119076371193, 1.5795024633407593, -0.2680153548717499, -0.015231357887387276, 0.6212132573127747, 1.0376043319702148, -0.7076330780982971, -0.14916926622390747, -0.8474617004394531, 0.638671338558197, -0.6297200322151184, 0.06733881682157516, -0.1078905388712883, -1.3372700214385986, 2.3791403770446777, 1.3534806966781616, 0.6537915468215942, -0.8210790753364563, -0.39746278524398804, -0.8431605696678162, 0.022339969873428345, 1.167518973350525, 0.2577197551727295, 0.6385740637779236, -1.1635583639144897, -0.016935672610998154, -0.9046909213066101, 0.1914374977350235, -0.8277912735939026, 0.03473489731550217, 0.445978581905365, -0.15936172008514404, 0.5023988485336304, 0.9685167074203491, 0.5251375436782837, -0.21976879239082336, -0.37468981742858887, -0.5958842635154724, -0.06166384369134903, -0.7576429843902588, -0.18045878410339355, 0.4579526484012604, -0.06546224653720856, 0.43490058183670044, 0.7938107252120972, 0.4344558119773865, -0.9002174735069275, 0.06239241361618042, 0.5299779772758484, -0.4090549051761627, 1.3338117599487305, 4.032529830932617, -0.41885724663734436, -0.9273133873939514, 0.7316491007804871, 0.598893940448761, -0.3389354944229126, -1.1082638502120972, 0.19746722280979156, -0.10217269510030746, -0.3144814372062683, 0.3396718502044678, 2.157517433166504, -1.003414511680603, -1.1999131441116333, 1.2112630605697632, 0.8285266757011414, -0.1915508508682251, 1.4793102741241455, -0.20172812044620514, 0.11118315905332565, 0.754660964012146, 0.3937753438949585, 0.3029715120792389, 0.9077270030975342, -0.7091923356056213, -0.3215942680835724, -0.5109661817550659, -0.052153442054986954, -0.900422990322113, -1.3986173868179321, -0.8953987956047058, 0.664132297039032, 0.26189863681793213, -1.0884209871292114, 0.7225176692008972, -0.6760314702987671, -1.1408898830413818, -2.053232192993164, -0.2553997337818146, 0.45973092317581177, -0.07582854479551315, -0.03398354351520538, -0.45875102281570435, 0.16597896814346313, -0.5867007374763489, -1.2276781797409058, 0.08464879542589188, 0.8852401375770569, -0.46610313653945923, 0.5943552255630493, 0.0833653137087822, 0.6171523332595825, -0.1614576280117035, -0.5549526214599609, 0.07708843052387238, -0.6521407961845398, -0.7752677798271179, -0.41582009196281433, 0.3822118043899536, 0.003933769650757313, 0.0344674251973629, -0.01891213469207287, 0.12470418214797974, -0.4443114697933197, -0.5364000797271729, -0.22637437283992767, -0.2846042513847351, 1.3546273708343506, -0.09433968365192413, 0.38637468218803406, -0.2513827383518219, 0.8655991554260254, 0.15730994939804077, 0.733832597732544, -0.5749954581260681, -0.6813852787017822, 0.6859632134437561, 0.8665807247161865, 0.08699018508195877, -0.7998385429382324, 0.8555545806884766, 1.385532021522522, 0.4699721932411194, 1.068503975868225, -0.031331490725278854, 0.31339526176452637, 0.36772891879081726, 0.28768661618232727, 1.6149373054504395, -1.296806812286377, -0.42283302545547485, -0.28493037819862366, -0.022600233554840088, 1.3545881509780884, -0.2340068221092224, 0.06214220076799393, -0.5010323524475098, -0.5321836471557617, 0.9646769165992737, -1.5075398683547974, 1.2526285648345947, 2.214531898498535, -0.9636873602867126, 0.7357030510902405, 1.4456325769424438, 0.2168843001127243, 0.7869573831558228, -0.13438114523887634, -1.1253690719604492, -0.5030725598335266, 1.5784825086593628, 0.8663685917854309, 0.22010380029678345, 1.2727991342544556, 0.10394527018070221, -0.8154317736625671, 0.0917787179350853, -0.21537460386753082, 0.22515447437763214, 0.14777997136116028, -0.1378639042377472, 0.054784346371889114, 0.7214288115501404, -0.4935878813266754, -0.13191579282283783, -0.24180248379707336, 0.8480783104896545, 0.5269560217857361, -1.0095725059509277, -0.8485996723175049, -0.3310410678386688, -1.7307696342468262, -0.5229339003562927, -0.3965691924095154, -0.1537267565727234, 0.1512121558189392, -0.9072139263153076, -0.5766147375106812, 0.9501603841781616, -1.0004891157150269, -2.0913405418395996, 0.9395622611045837, -0.6602624654769897, 0.9185859560966492, -0.4308270812034607, -0.4183182716369629, -0.3987850844860077, 0.855529248714447, -0.4317927658557892, 0.013379105366766453, 0.25641006231307983, 0.40496841073036194, -0.4951484799385071, 0.2948271334171295, -0.8683901429176331, -0.8149306178092957, -0.03279824182391167, 1.2475032806396484, 0.19749107956886292, -2.833357334136963, -0.13681809604167938, 2.5671558380126953, -1.0553900003433228, 0.36904987692832947, 2.593820571899414, -0.8598019480705261, 0.24499471485614777, 0.9542664289474487, 1.5621610879898071, -0.24799533188343048, 0.805902898311615, 0.22173674404621124, 0.5275965929031372, -0.17625494301319122, 0.7533872723579407, -1.6283048391342163, 0.7016459107398987, -0.4515664875507355, -0.28366273641586304, -0.7657015323638916, -0.25016313791275024, -1.088510513305664, 0.38472744822502136, 0.6769755482673645, 0.3676595091819763, 0.46455928683280945, 0.039463020861148834, 0.4587879478931427, -0.8108278512954712, 0.2598423957824707, 0.8881469964981079, -0.2924939692020416, -0.42665061354637146, -0.4089321196079254, 0.36605873703956604, -1.270892858505249, 0.08878681808710098, -0.8626613020896912, 0.2744051218032837, 0.8229535222053528, -0.4679422080516815, 1.424270510673523, -0.6986895799636841, -1.4346706867218018, -2.3086764812469482, -1.0041663646697998, 1.0205570459365845, 0.6137961745262146, -0.41273924708366394, 0.23465439677238464, 1.7142930030822754, -0.2517821490764618, -0.35666990280151367, -0.8046519160270691, -0.22130607068538666, -0.011488537304103374, 1.2828510999679565, 0.24674326181411743, 0.07552424818277359, 0.300187885761261, 1.1933907270431519, 0.09056577831506729, 0.6955752372741699, 1.5163812637329102, -0.6374694108963013, -0.2591145932674408, -0.14732594788074493, -0.7316429018974304, -1.0586882829666138, -1.962838888168335, -0.8305853009223938, -0.25581005215644836, 0.5952919721603394, -0.10122887790203094, 0.37867972254753113, 1.3561429977416992, 0.32374584674835205, 1.0768733024597168, -1.1027733087539673, 0.25966936349868774, -1.0027782917022705, -0.2317354381084442, 0.13147157430648804, 0.6731062531471252, 0.9311864972114563, 0.16078782081604004, -0.06347150355577469, -1.127803921699524, 1.0859230756759644, -0.05299338698387146, -0.0916958749294281, -0.1152501106262207, -0.1876542866230011, 0.961541473865509, 0.3414599597454071, -0.4458370804786682, -0.6940256357192993, 0.2692474126815796, 0.6449998021125793, 0.5257284641265869, -1.2579660415649414, 0.06764481961727142, -1.0539761781692505, 0.152058944106102, 0.5853472352027893, 0.18800219893455505, 0.6064897775650024, -0.48305991291999817, 0.7645924091339111, 0.519838273525238, -0.5374780297279358, -0.48863285779953003, 1.5892449617385864, -1.0736191272735596, -0.8423982262611389, 0.5304821729660034, -0.30236586928367615, -0.2331259399652481, -1.402252197265625, 1.0032541751861572, 0.25986960530281067, 0.3195355534553528, 0.6663399338722229, 1.469584345817566, -0.25291943550109863, 0.5780280232429504, -0.4923444092273712, 0.9900853037834167, -1.9150124788284302, 0.2608342170715332, 0.09661253541707993, 0.33580395579338074, 0.44209161400794983, 1.3630222082138062, 0.027984609827399254, -0.07663511484861374, 1.2112826108932495, 1.3055411577224731, 0.7223176956176758, 0.7741977572441101, -0.012070536613464355, 0.792019248008728, -0.8358588218688965, -0.24300603568553925, 1.4761286973953247, 0.47833067178726196, -0.862320601940155, 1.6347403526306152, -0.044975198805332184, 0.23384100198745728, 0.05975710228085518, 0.5373275279998779, -0.6036093831062317, -0.9584037065505981, -0.3424752950668335, 1.033970832824707, -0.9000636339187622, -1.7259516716003418, 0.09281637519598007, 0.5703420042991638, 0.27247151732444763, -0.6554797887802124, 1.29293954372406, -0.8269041180610657, -1.414451003074646, 0.2098378837108612, -1.495086669921875, 0.5303477048873901, -0.42576688528060913, -1.0140630006790161, 0.3031359910964966, 1.0610772371292114, 1.1370598077774048, -0.06484775245189667, -0.4315686523914337, 0.2688913345336914, -0.23305165767669678, 0.33030998706817627, 0.8156024217605591, -1.7741761207580566, -1.0613518953323364, -0.5163298845291138, 0.967062771320343, 0.3016012907028198, 0.7268349528312683, -1.3398020267486572, 0.2719517946243286, 1.6705362796783447, 0.262799471616745, 0.5916933417320251, -0.9717416763305664, 0.7251421809196472, -0.48113521933555603, 0.3160078823566437, 0.455743670463562, 1.3180030584335327, 1.0612077713012695, 0.5005386471748352, 0.4365060031414032, 0.17243944108486176, 0.46606460213661194, 0.5488066077232361, 1.0039429664611816, 0.5314913988113403, 0.8226202130317688, -0.16402140259742737, -0.29089078307151794, -0.737926721572876, 0.17781904339790344, -0.23646672070026398, -2.842583179473877, 0.4516940712928772, 0.7915653586387634, 0.31149381399154663, -0.21878674626350403, 0.946761965751648, -1.4096537828445435, 0.8708169460296631, -0.5886717438697815, -0.37809890508651733, 1.6523891687393188, -0.4132094383239746, -0.18555893003940582, 0.9703131318092346, 0.17997977137565613, 0.23230880498886108, 0.8182770609855652, -0.5146350860595703, -0.28205400705337524, 0.7836549282073975, 0.29722511768341064, 1.1084295511245728, -0.11518625169992447, -0.697401762008667, 1.015866756439209, 0.4268779456615448, 0.22288118302822113, 0.9527218341827393, -0.76492840051651, 1.0399131774902344, -0.20303545892238617, 0.5362374186515808, 1.086016297340393, -0.19128958880901337, 0.24724100530147552, 0.07818453758955002, -0.6610020995140076, 1.923974871635437, 0.2581028640270233, 1.200670838356018, -0.6744351983070374, 0.24254198372364044, 1.6161164045333862, -0.3558301031589508, -1.7124642133712769, 0.23828400671482086, -1.3082640171051025, -1.012444257736206, 1.2147713899612427, 1.4965678453445435, 0.43377625942230225, -0.5428401827812195, 0.2039150446653366, -0.3645533323287964, 0.08567041903734207, 0.09551329165697098, 0.6319390535354614, 1.0147488117218018, 0.9743899703025818, -0.3659929633140564, -0.7726812958717346, 1.8289811611175537, 0.5826027989387512, -0.2755247950553894, 0.7164722084999084, 1.0314130783081055, 0.810772180557251, -0.43021267652511597, 1.7385824918746948, -0.2942686676979065, 0.122498519718647, -0.3574557304382324, 0.4564189612865448, 0.684938907623291, 0.6730185151100159, 0.08213746547698975, 0.7038066983222961, 0.6101676225662231, -0.3290257751941681, 0.94777911901474, 0.2292790412902832, 0.3792678415775299, 0.4460514485836029, 0.5465898513793945, 0.4584771990776062, -0.9808205366134644, 0.6438464522361755, -0.25934723019599915, 0.3816877007484436, -1.8941516876220703, 1.8761612176895142, 1.3476364612579346, -0.371740460395813, -0.17359425127506256, -0.28092697262763977, 0.9289122819900513, 0.8635220527648926, 1.1393611431121826, -0.3358951508998871, -0.09807706624269485, -0.3378733694553375, -0.7841969132423401, -1.2873581647872925, -0.30210500955581665, -1.2338502407073975, 0.2763659954071045, 0.774817705154419, -0.6226937770843506, -1.059740424156189, 0.8003582954406738, -1.0926438570022583, -0.7416285276412964, 1.1888731718063354, -0.5579878687858582, -0.5636143088340759, -0.6792082190513611, 0.66744464635849, -0.06904187053442001, -1.224100947380066, 0.8757246732711792, 0.5300902128219604, -0.2684779465198517, -0.14843402802944183, 0.8857123851776123, 0.6778329014778137, -0.13366013765335083, 0.18218138813972473, -1.8435760736465454, -0.2582884728908539, 0.8576270937919617, 0.5240642428398132, -0.2218858003616333, -2.3227620124816895, 0.32481271028518677, 0.3308824598789215, -0.12319739162921906, -1.1492794752120972, 0.7900943756103516, 0.2672337293624878, -0.5655728578567505, -0.396100252866745, 1.0296437740325928, 1.6514372825622559, -0.4066377878189087, 1.0908482074737549, 1.3751184940338135, -0.8535380959510803, 0.5225468277931213, -0.0006373909418471158, 0.6404005885124207, -0.8335983157157898, -0.30718767642974854, 0.9382672905921936, -1.4374334812164307, 0.3405800461769104, -1.8749080896377563, -0.015412780456244946, 0.9586003422737122, 0.9749032855033875, 0.4089370667934418, 0.3300617039203644, -0.2936447262763977, -1.5275622606277466, -1.215834140777588, -0.09590303152799606, -0.3736042380332947, -0.4046858847141266, -0.7604740858078003, 0.30732089281082153, 0.9786489605903625, -0.024486642330884933, -0.44024211168289185, -0.5303723216056824, -0.3100026845932007, 0.41158047318458557, 0.08978889137506485, -1.1836999654769897, 1.1328039169311523, -1.8808642625808716, -2.6324923038482666, 0.49613094329833984, 0.7676315307617188, -0.81942218542099, 1.4463374614715576, 0.8461065888404846, 1.2071186304092407, -0.7644268870353699, -0.7974808216094971, -0.6074591279029846, 0.22461646795272827, -0.03352367877960205, 0.5056027770042419, -0.2354835569858551, 0.7848227024078369, -1.4730831384658813, -0.28553497791290283, -0.02128131315112114, 1.7117793560028076, 0.5307771563529968, -1.0378741025924683, 0.5511612296104431, -0.34776780009269714, 0.3234902620315552, 0.018385043367743492, 1.9667503833770752, -0.3220098316669464, 1.5246467590332031, 1.5930222272872925, 0.5382782816886902, 0.3997541666030884, -0.029441697522997856, 1.164792776107788, 1.0483663082122803, 0.6762118935585022, 0.2721809446811676, 1.5836933851242065, -0.36960771679878235, -0.30111125111579895, 1.0387370586395264, -0.8230786919593811, 0.02849493734538555, -0.401652455329895, 3.151557207107544, 1.271631121635437, 1.051163911819458, -0.4742489755153656, -0.13422998785972595, -1.0860100984573364, -0.2242734283208847, 1.555233359336853, -1.684885025024414, -0.5929245352745056, -0.07701154053211212, 1.0613621473312378, -0.3573145866394043, -0.16633635759353638, -1.023155927658081, 0.3753410577774048, 0.9092200994491577, 0.8633676767349243, 1.3486155271530151, 0.48645275831222534, 0.7589786648750305, 1.6779955625534058, 1.0541821718215942, -0.36466947197914124, 0.3677104115486145, -1.553744912147522, 0.41659897565841675, 0.3703010678291321, 0.6729130744934082, 0.11305806785821915, 0.04704665765166283, 1.4834892749786377, -0.5027344822883606, -0.06039338931441307, -0.30485206842422485, 0.5961495637893677, 1.049264669418335, -0.6112093329429626, 0.4830496907234192, -1.5174516439437866, -0.21395312249660492, -0.2307266891002655, 0.21985045075416565, -0.06488046050071716, 2.468750238418579, 0.04897520691156387, 0.33386287093162537, 0.17066818475723267, -0.058882396668195724, -0.5094000697135925, 0.1412844955921173, -0.9301515221595764, -1.0718261003494263, 0.6923262476921082, 1.2468719482421875, -0.873746395111084, 1.0347999334335327, 0.042773518711328506, 0.4812251627445221, -0.4102485477924347, 0.1207783967256546, 0.6320503354072571, 0.9662494659423828, 0.4326235353946686, -0.5337560772895813, 0.5703607797622681, -0.2519015073776245, -0.4809989929199219, -0.8399595022201538, 0.5395193696022034, 1.3570252656936646, 0.33629295229911804, -1.0144877433776855, 0.4141647517681122, -0.3494333028793335, -0.39241519570350647, 0.26294541358947754, -0.9430385231971741, 0.1588028371334076, -1.2582104206085205, -0.9495152235031128, -0.2437324821949005, -0.10070988535881042, 0.8682916760444641, 0.5922117233276367, 0.9088482856750488, 1.2485711574554443, -0.670518696308136, -0.8908605575561523, 0.24662235379219055, 0.372570663690567, 0.8114488124847412, 0.958018958568573, 0.036077287048101425, 0.8445289134979248, 1.0080205202102661, 0.18947386741638184, 0.24260398745536804, 0.4650104343891144, -1.0607584714889526, 1.3718576431274414, -0.6729465126991272, 0.06432390213012695, -0.7248271703720093, -0.18928764760494232, 2.2607553005218506, -0.4354011118412018, -0.1217154935002327, 0.8562799096107483, 0.4336405396461487, -0.8363307118415833, 0.1935807168483734, 0.15258289873600006, 0.7323409914970398, -0.5788915753364563, 0.5015891194343567, -0.6020100712776184, 0.22939720749855042, -0.1671215146780014, 0.8100032210350037, 1.4101309776306152, -0.41793686151504517, 1.3974922895431519, -0.2698364853858948, -0.3047636151313782, -0.7949076890945435, 0.43689581751823425, 0.8589152097702026, 0.09348166733980179, -0.39030686020851135, 0.7240561842918396, 1.543845534324646, 0.2918202877044678, 1.050837755203247, -0.26964637637138367, 0.12725426256656647, 1.0904412269592285, 0.9542199969291687, 0.5019722580909729, -0.5938015580177307, -0.0776636153459549, 1.5826146602630615, -0.018145358189940453, -1.0955172777175903, 0.11015670001506805, -1.004249930381775, 0.49658170342445374, 0.8364162445068359, 0.3150012791156769, -0.40435969829559326, 1.656010389328003, -0.43965238332748413, 1.8304469585418701, 1.5268043279647827, -0.8604661822319031, 1.0329275131225586, 0.5303273797035217, 0.402413547039032, 0.20498813688755035, 0.019901921972632408, -0.5533758997917175, 0.2398901879787445, 1.0117279291152954, 0.03756370022892952, -0.11425872147083282, -0.42754608392715454, -0.5956221222877502, 1.2829158306121826, -0.020011141896247864, -0.010008986108005047, 0.04418850690126419, 1.2053484916687012, 0.5647292733192444, -0.26850128173828125, 0.7562182545661926, 0.9609339833259583, 0.7130849361419678, -0.4688777029514313, -0.7263156771659851, -0.3216785788536072, -0.3782753646373749, -0.4543387293815613, 1.0923529863357544, -0.7255597710609436, 0.1579989343881607, 1.393275499343872, 0.4758467674255371, -0.6269780397415161, 0.555738091468811, 0.07791661471128464, 1.115803837776184, -0.16730351746082306, -1.7548264265060425, -1.0915530920028687, 0.3272109627723694, 0.7198787331581116, 0.15869252383708954, 1.5325491428375244, 0.6273746490478516, -0.22920212149620056, 0.6271662712097168, 1.1550211906433105, 0.022395849227905273, 1.548167109489441, 0.05239686369895935, 0.10679496079683304, -1.2490460872650146, 0.5457632541656494, -1.3255480527877808, 0.1094525009393692, -0.4470067322254181, 0.10157430171966553, 0.49657270312309265, -0.3422251045703888, -0.3428717255592346, -1.0989112854003906, 0.7292521595954895, 0.10979542881250381, 1.0745795965194702, 1.4582161903381348, 0.2624737024307251, 0.24237756431102753, 1.0911977291107178, 0.24272166192531586, -0.31852635741233826, 0.9282757639884949, 0.17004258930683136, -1.4414691925048828, -0.6943944692611694, 0.5762156844139099, 0.13523069024085999, -0.07928363978862762, -1.3160996437072754, -0.7508617639541626, -0.3963170647621155, 0.8717027902603149, 0.6616354584693909, -0.7284257411956787, 0.8170806765556335, -0.12391214072704315, -1.011511206626892, -0.16653229296207428, -1.392063856124878, 1.456265926361084, 1.270193338394165, -0.04346597194671631, -0.2501539885997772, 1.2999532222747803, -0.2635824680328369, 0.8484753966331482, 0.5765373706817627, 0.2439611703157425, 0.405848890542984, 1.8943120241165161, 1.0930464267730713, 0.47026124596595764, -0.2158227413892746, 0.5740467309951782, -0.40225711464881897, 0.5754817724227905, 0.7490095496177673, 0.6491059064865112, 0.0867711529135704, 0.579806923866272, 0.8394874930381775, -0.9076231718063354, -0.11690628528594971, -0.5144726037979126, 0.18478679656982422, 0.8861205577850342, 0.5507660508155823, 0.062138043344020844, 0.0859030932188034, 1.1013994216918945, 0.7780859470367432, -1.508596420288086, 1.3157068490982056, -0.02061065472662449, 0.4520299732685089, 0.24951553344726562, 0.8582152724266052, -1.3059518337249756, 0.5435992479324341, 0.9995498657226562, 0.8687682747840881, 1.1293929815292358, 0.1182815358042717, -0.5584964156150818, -0.0911945104598999, -2.492168426513672, -0.3337835669517517, 0.9179407358169556, 0.12108004093170166, 0.39262089133262634, 0.3834787607192993, 0.14400067925453186, 0.6127351522445679, 1.0006606578826904, -0.3388870656490326, 1.3470948934555054, 0.3191034495830536, 0.6416683197021484, 2.405911445617676, 1.0034737586975098, 1.7496037483215332, -0.4447437524795532, -0.08392654359340668, -0.4861396849155426, 0.9100956320762634, -0.39819446206092834, -1.0313358306884766, -0.7500005960464478, -0.0474570132791996, 0.6214759349822998, 0.44665735960006714, 0.939668595790863, -0.4448482096195221, -0.33317795395851135, -0.6649143099784851, 0.9275359511375427, 1.1091188192367554, 0.6849321126937866, -1.1289687156677246, 0.6398919820785522, -0.4811200499534607, 0.05386725440621376, 0.19938969612121582, -1.1293327808380127, -0.5220584273338318, -1.3109878301620483, 0.5708319544792175, 0.053319286555051804, -0.5342313647270203, 0.20898854732513428, -0.4800363779067993, -0.4249866306781769, 0.9517281651496887, 1.534987211227417, 0.3781431019306183, 0.8037827610969543, 0.5449170470237732, 1.7283538579940796, 1.5257841348648071, -0.05732261389493942, 0.14214007556438446, -0.06239773705601692, 0.5113009810447693, 1.3664219379425049, 1.1022363901138306, 0.2600550055503845, 0.5225167274475098, 0.36129099130630493, 1.1423321962356567, 0.1436927318572998, 0.21705111861228943, 0.5551620125770569, 1.6153665781021118, -0.6808907389640808, 0.018261197954416275, -0.37196415662765503, 0.7084493637084961, 1.1560944318771362, 0.34042492508888245, -0.11767666041851044, 0.6521451473236084, 0.7500288486480713, 0.6613458395004272, 1.5875478982925415, 1.660003900527954, -0.1457277238368988, -2.253944158554077, 0.8130041360855103, 0.27323296666145325, -1.1058746576309204, 0.36612051725387573, -1.3572407960891724, 0.7620600461959839, 0.8414766788482666, 0.8342466950416565, 0.9264625906944275, -0.3163425624370575, 0.02923250012099743, 0.5840960144996643, 0.07232964038848877, -0.6192368268966675, -0.37411195039749146, 1.1144832372665405, -1.778894305229187, 0.01097877323627472, 0.2337062507867813, 1.4407787322998047, 0.5385745167732239, 0.5460538268089294, 0.30779704451560974, 0.9452893137931824, -0.34561631083488464, 0.259483277797699, 1.256514549255371, -0.8379605412483215, 0.9836186766624451, 0.24784961342811584, -0.2636171877384186, 0.4639705419540405, -0.6606394052505493, 0.2619413733482361, 1.0670605897903442, 0.48818209767341614, -0.4376562535762787, 1.2261545658111572, 0.9127340316772461, -1.1223658323287964, -0.08234243839979172, 0.3134010136127472, 0.29115375876426697, 1.1298779249191284, -0.5173208117485046, 1.3418399095535278, -0.45205560326576233, -0.9484046697616577, -0.5165708065032959, 0.6530460715293884, 0.33672741055488586, 0.3465670943260193, 0.36379095911979675, 1.6053006649017334, 0.9748526215553284, 0.21044769883155823, -0.4970995783805847, -0.4820665419101715, -0.04381295666098595, -0.16861560940742493, -0.7231128811836243, -0.20925384759902954, -0.6498520970344543, -0.35963231325149536, 1.0138930082321167, 0.04430055618286133, 0.4668925106525421, 1.7282167673110962, -0.6902900338172913, -0.2984910011291504, -0.4804086685180664, 1.0898659229278564, 0.059377074241638184, 0.7230729460716248, -0.6926661133766174, 1.3298612833023071, -0.605229377746582, 1.671138882637024, -0.20532935857772827, 0.19402097165584564, -0.3413960337638855, -0.07963846623897552, -0.870858907699585, -0.296911358833313, 0.7758028507232666, 0.9004979133605957, 0.22393648326396942, 0.4137844145298004, 2.0985944271087646, 0.33040085434913635, 1.0561847686767578, -1.1144729852676392, 0.04916185885667801, 0.4571821093559265, 0.27152109146118164, -0.15758340060710907, 0.28915780782699585, 0.8348183035850525, -1.2559332847595215, -1.1639423370361328, 0.6882803440093994, 0.4025547504425049, -0.057504989206790924, 1.493494987487793, -0.37739214301109314, -0.2692364454269409, -0.04179150611162186, -0.4434122145175934, 1.1402454376220703, -0.7770587801933289, 0.5020185112953186, 0.7832685112953186, 0.23953457176685333, 1.1034878492355347, -0.12874503433704376, 1.0245857238769531, -0.5754788517951965, 0.09188815206289291, 1.1028674840927124, -0.07253583520650864, -0.3791196346282959, -0.8315320014953613, 0.757881224155426, 0.8148852586746216, -0.06428422033786774, -0.0865720808506012], "yaxis": "y"}],
                        {"legend": {"itemsizing": "constant", "tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "white", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "white", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "#C8D4E3", "linecolor": "#C8D4E3", "minorgridcolor": "#C8D4E3", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "#C8D4E3", "linecolor": "#C8D4E3", "minorgridcolor": "#C8D4E3", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "white", "showlakes": true, "showland": true, "subunitcolor": "#C8D4E3"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "white", "polar": {"angularaxis": {"gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": ""}, "bgcolor": "white", "radialaxis": {"gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "white", "gridcolor": "#DFE8F3", "gridwidth": 2, "linecolor": "#EBF0F8", "showbackground": true, "ticks": "", "zerolinecolor": "#EBF0F8"}, "yaxis": {"backgroundcolor": "white", "gridcolor": "#DFE8F3", "gridwidth": 2, "linecolor": "#EBF0F8", "showbackground": true, "ticks": "", "zerolinecolor": "#EBF0F8"}, "zaxis": {"backgroundcolor": "white", "gridcolor": "#DFE8F3", "gridwidth": 2, "linecolor": "#EBF0F8", "showbackground": true, "ticks": "", "zerolinecolor": "#EBF0F8"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""}, "baxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""}, "bgcolor": "white", "caxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#EBF0F8", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#EBF0F8", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x0"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "x1"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('419c732f-dc67-4f4d-853a-02bb5e04291d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


The graph is stretched horizontally and slightly vertically. By hovering the cursor on some of the points, we see `reportedly`, `myanmar`, `rohingya`, `trumps`, `donald`, `barack` are all stronger indicators for whether a news is fake or not. 
