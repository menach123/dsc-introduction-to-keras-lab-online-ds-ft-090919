
# Keras - Lab

## Introduction

In this lab you'll once again build a neural network, but this time you will be using Keras to do a lot of the heavy lifting.


## Objectives

You will be able to:

- Build a neural network using Keras 
- Evaluate performance of a neural network using Keras 

## Required Packages

We'll start by importing all of the required packages and classes.


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import optimizers
import warnings
warnings.filterwarnings('ignore')

```

    Using TensorFlow backend.
    

## Load the data

In this lab you will be classifying bank complaints available in the `'Bank_complaints.csv'` file. 


```python
# Import data
df = pd.read_csv('Bank_complaints.csv')

# Inspect data
print(df.info())
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 60000 entries, 0 to 59999
    Data columns (total 2 columns):
    Product                         60000 non-null object
    Consumer complaint narrative    60000 non-null object
    dtypes: object(2)
    memory usage: 937.6+ KB
    None
    




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
      <th>Product</th>
      <th>Consumer complaint narrative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Student loan</td>
      <td>In XX/XX/XXXX I filled out the Fedlaon applica...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Student loan</td>
      <td>I am being contacted by a debt collector for p...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Student loan</td>
      <td>I cosigned XXXX student loans at SallieMae for...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Student loan</td>
      <td>Navient has sytematically and illegally failed...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Student loan</td>
      <td>My wife became eligible for XXXX Loan Forgiven...</td>
    </tr>
  </tbody>
</table>
</div>



As mentioned earlier, your task is to categorize banking complaints into various predefined categories. Preview what these categories are and what percent of the complaints each accounts for. 


```python
df['Product'].unique()
```




    array(['Student loan', 'Credit card', 'Bank account or service',
           'Consumer Loan', 'Checking or savings account', 'Mortgage',
           'Credit reporting'], dtype=object)



## Preprocessing

Before we build our neural network, we need to do several preprocessing steps. First, we will create word vector counts (a bag of words type representation) of our complaints text. Next, we will change the category labels to integers. Finally, we will perform our usual train-test split before building and training our neural network using Keras. With that, let's start munging our data! 

## One-hot encoding of the complaints

Our first step again is to transform our textual data into a numerical representation. As we saw in some of our previous lessons on NLP, there are many ways to do this. Here, we'll use the `Tokenizer()` class from the `preprocessing.text` sub-module of the Keras package.   

As with our previous work using NLTK, this will transform our text complaints into word vectors. (Note that the method of creating a vector is different from our previous work with NLTK; as you'll see, word order will be preserved as opposed to a bag of words representation). In the below code, we'll only keep the 2,000 most common words and use one-hot encoding. 




```python
# ⏰ This cell may take about thirty seconds to run

# Raw text complaints
complaints = df['Consumer complaint narrative'] 

# Initialize a tokenizer 
tokenizer = Tokenizer(num_words=2000) 

# Fit it to the complaints
tokenizer.fit_on_texts(complaints) 

# Generate sequences
sequences = tokenizer.texts_to_sequences(complaints) 
print('sequences type:', type(sequences))

# Similar to sequences, but returns a numpy array
one_hot_results= tokenizer.texts_to_matrix(complaints, mode='binary') 
print('one_hot_results type:', type(one_hot_results))

# Useful if we wish to decode (more explanation below)
word_index = tokenizer.word_index 

# Tokens are the number of unique words across the corpus
print('Found %s unique tokens.' % len(word_index)) 

# Our coded data
print('Dimensions of our coded results:', np.shape(one_hot_results)) 
```

    sequences type: <class 'list'>
    one_hot_results type: <class 'numpy.ndarray'>
    Found 50334 unique tokens.
    Dimensions of our coded results: (60000, 2000)
    

## Decoding Word Vectors 

As a note, you can also decode these vectorized representations of the reviews. The `word_index` variable, defined above, stores the mapping from the label number to the actual word. Somewhat tediously, we can turn this dictionary inside out and map it back to our word vectors, giving us roughly the original complaint back. (As you'll see, the text won't be identical as we limited ourselves to top 2000 words.)

## Python Review / Mini Challenge

While a bit tangential to our main topic of interest, we need to reverse our current dictionary `word_index` which maps words from our corpus to integers. In decoding our `one_hot_results`, we will need to create a dictionary of these integers to the original words. Below, take the `word_index` dictionary object and change the orientation so that the values are keys and the keys values. In other words, you are transforming something of the form {A:1, B:2, C:3} to {1:A, 2:B, 3:C}. 


```python
reverse_index = {word_index[i]:i for i in word_index}
```

## Back to Decoding Our Word Vectors...


```python
comment_idx_to_preview = 19
print('Original complaint text:')
print(complaints[comment_idx_to_preview])
print('\n\n')

# The reverse_index cell block above must be complete in order for this cell block to successively execute 
decoded_review = ' '.join([reverse_index.get(i) for i in sequences[comment_idx_to_preview]])
print('Decoded review from Tokenizer:')
print(decoded_review)
```

    Original complaint text:
    I have already filed several complaints about AES/PHEAA. I was notified by a XXXX XXXX let @ XXXX, who pretended to be from your office, he said he was from CFPB. I found out this morning he is n't from your office, but is actually works at XXXX. 
    
    This has wasted weeks of my time. They AES/PHEAA confirmed and admitted ( see attached transcript of XXXX, conversation at XXXX ( XXXX ) with XXXX that proves they verified the loans are not mine ) the student loans they had XXXX, and collected on, and reported negate credit reporting in my name are in fact, not mine. 
    They conclued their investigation on XXXX admitting they made a mistake and have my name on soneone elses loans. I these XXXX loans total {$10000.00}, original amount. My XXXX loans I got was total {$3500.00}. We proved by providing AES/PHEAA, this with my original promissary notes I located recently, the XXXX of my college provided AES/PHEAA with their original shoeinf amounts of my XXXX loans which show different dates and amounts, the dates and amounts are not even close to matching these loans they have in my name, The original lender, XXXX XXXX Bank notifying AES/PHEAA, they never issued me a student loan, and original Loan Guarantor, XXXX, notifying AES/PHEAA, they never were guarantor of my loans. 
    
    XXXX straight forward. But today, this person, XXXX XXXX, told me they know these loans are not mine, and they refuse to remove my name off these XXXX loan 's and correct their mistake, essentially forcing me to pay these loans off, bucause in XXXX they sold the loans to XXXX loans. 
    
    This is absurd, first protruding to be this office, and then refusing to correct their mistake. 
    
    Please for the love of XXXX will soneone from your office call me at XXXX, today. I am a XXXX vet and they are knowingly discriminating against me. 
    Pretending to be you.
    
    
    
    Decoded review from Tokenizer:
     to be you
    

## Convert the Products to Numerical Categories

On to step two of our preprocessing: converting our descriptive categories into integers.


```python
product = df['Product']

# Initialize
le = preprocessing.LabelEncoder() 
le.fit(product)
print('Original class labels:')
print(list(le.classes_))
print('\n')
product_cat = le.transform(product)  

# If you wish to retrieve the original descriptive labels post production
# list(le.inverse_transform([0, 1, 3, 3, 0, 6, 4])) 

print('New product labels:')
print(product_cat)
print('\n')

# Each row will be all zeros except for the category for that observation 
print('One hot labels; 7 binary columns, one for each of the categories.') 
product_onehot = to_categorical(product_cat)
print(product_onehot)
print('\n')

print('One hot labels shape:')
print(np.shape(product_onehot))
```

    Original class labels:
    ['Bank account or service', 'Checking or savings account', 'Consumer Loan', 'Credit card', 'Credit reporting', 'Mortgage', 'Student loan']
    
    
    New product labels:
    [6 6 6 ... 4 4 4]
    
    
    One hot labels; 7 binary columns, one for each of the categories.
    [[0. 0. 0. ... 0. 0. 1.]
     [0. 0. 0. ... 0. 0. 1.]
     [0. 0. 0. ... 0. 0. 1.]
     ...
     [0. 0. 0. ... 1. 0. 0.]
     [0. 0. 0. ... 1. 0. 0.]
     [0. 0. 0. ... 1. 0. 0.]]
    
    
    One hot labels shape:
    (60000, 7)
    

## Train-test split

Now for our final preprocessing step: the usual train-test split. 


```python
random.seed(123)
test_index = random.sample(range(1,60000), 1500)

test = one_hot_results[test_index]
train = np.delete(one_hot_results, test_index, 0)

label_test = product_onehot[test_index]
label_train = np.delete(product_onehot, test_index, 0)

print('Test label shape:', np.shape(label_test))
print('Train label shape:', np.shape(label_train))
print('Test shape:', np.shape(test))
print('Train shape:', np.shape(train))
```

    Test label shape: (1500, 7)
    Train label shape: (58500, 7)
    Test shape: (1500, 2000)
    Train shape: (58500, 2000)
    

## Building the network

Let's build a fully connected (Dense) layer network with relu activation in Keras. You can do this using: `Dense(16, activation='relu')`. 

In this example, use two hidden layers with 50 units in the first layer and 25 in the second, both with a `'relu'` activation function. Because we are dealing with a multiclass problem (classifying the complaints into 7 categories), we use a use a `'softmax'` classifier in order to output 7 class probabilities per case.  


```python
# Initialize a sequential model
model = models.Sequential()

# Two layers with relu activation
units = 50
activation = 'relu'
model.add(layers.Dense(units, activation='relu'))
units = 25
model.add(layers.Dense(units, activation='relu'))
# One layer with softmax activation 
units = 7
model.add(layers.Dense(units, activation='softmax'))
```

## Compiling the model

Now, compile the model! This time, use `'categorical_crossentropy'` as the loss function and stochastic gradient descent, `'SGD'` as the optimizer. As in the previous lesson, include the accuracy as a metric.


```python
# Compile the model
model.compile(optimizer=optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

```

## Training the model

In the compiler, you'll be passing the optimizer (SGD = stochastic gradient descent), loss function, and metrics. Train the model for 120 epochs in mini-batches of 256 samples.

_Note:_ ⏰ _Your code may take about one to two minutes to run._


```python
# Train the model 
history = model.fit(train,
                    label_train,
                    epochs=120,
                    batch_size=256)

```

    Epoch 1/120
    58500/58500 [==============================] - 2s 37us/step - loss: 1.8300 - acc: 0.2583
    Epoch 2/120
    58500/58500 [==============================] - 2s 35us/step - loss: 1.4783 - acc: 0.4887
    Epoch 3/120
    58500/58500 [==============================] - 2s 35us/step - loss: 1.1572 - acc: 0.6212
    Epoch 4/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.9419 - acc: 0.6964
    Epoch 5/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.8138 - acc: 0.7249
    Epoch 6/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.7384 - acc: 0.7417
    Epoch 7/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.6907 - acc: 0.7537
    Epoch 8/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.6574 - acc: 0.7624
    Epoch 9/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.6327 - acc: 0.7706
    Epoch 10/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.6126 - acc: 0.7775
    Epoch 11/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.5959 - acc: 0.7835
    Epoch 12/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.5815 - acc: 0.7882
    Epoch 13/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.5687 - acc: 0.7936
    Epoch 14/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.5575 - acc: 0.7978
    Epoch 15/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.5469 - acc: 0.8022
    Epoch 16/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.5376 - acc: 0.8059
    Epoch 17/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.5291 - acc: 0.8091
    Epoch 18/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.5211 - acc: 0.8121
    Epoch 19/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.5138 - acc: 0.8148
    Epoch 20/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.5069 - acc: 0.8184
    Epoch 21/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.5006 - acc: 0.8205
    Epoch 22/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4942 - acc: 0.8232
    Epoch 23/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4887 - acc: 0.8253
    Epoch 24/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4832 - acc: 0.8275
    Epoch 25/120
    58500/58500 [==============================] - 2s 36us/step - loss: 0.4785 - acc: 0.8295
    Epoch 26/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4738 - acc: 0.8314
    Epoch 27/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4695 - acc: 0.8330
    Epoch 28/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4650 - acc: 0.8346
    Epoch 29/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4610 - acc: 0.8360
    Epoch 30/120
    58500/58500 [==============================] - 2s 36us/step - loss: 0.4569 - acc: 0.8375
    Epoch 31/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4533 - acc: 0.8389
    Epoch 32/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4499 - acc: 0.8407
    Epoch 33/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4466 - acc: 0.8411
    Epoch 34/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4432 - acc: 0.8425
    Epoch 35/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4402 - acc: 0.8431
    Epoch 36/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4370 - acc: 0.8442
    Epoch 37/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4345 - acc: 0.8457
    Epoch 38/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4315 - acc: 0.8468
    Epoch 39/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4289 - acc: 0.8473
    Epoch 40/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4265 - acc: 0.8492
    Epoch 41/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4238 - acc: 0.8502
    Epoch 42/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4217 - acc: 0.8505
    Epoch 43/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4192 - acc: 0.8519
    Epoch 44/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4171 - acc: 0.8526
    Epoch 45/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4150 - acc: 0.8537
    Epoch 46/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4128 - acc: 0.8537
    Epoch 47/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4106 - acc: 0.8552
    Epoch 48/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4088 - acc: 0.8557
    Epoch 49/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4066 - acc: 0.8560
    Epoch 50/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4048 - acc: 0.8580
    Epoch 51/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.4031 - acc: 0.8573
    Epoch 52/120
    58500/58500 [==============================] - 2s 36us/step - loss: 0.4012 - acc: 0.8585
    Epoch 53/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.3996 - acc: 0.8596
    Epoch 54/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3974 - acc: 0.8600
    Epoch 55/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3958 - acc: 0.8606
    Epoch 56/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3943 - acc: 0.8608
    Epoch 57/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.3930 - acc: 0.8619
    Epoch 58/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3915 - acc: 0.8621
    Epoch 59/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3897 - acc: 0.8627
    Epoch 60/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3883 - acc: 0.8635
    Epoch 61/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3867 - acc: 0.8635
    Epoch 62/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3851 - acc: 0.8647
    Epoch 63/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3838 - acc: 0.8648
    Epoch 64/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3828 - acc: 0.8649
    Epoch 65/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3812 - acc: 0.8653
    Epoch 66/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3798 - acc: 0.8669
    Epoch 67/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3784 - acc: 0.8662
    Epoch 68/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3776 - acc: 0.8670
    Epoch 69/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3760 - acc: 0.8668
    Epoch 70/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3745 - acc: 0.8671
    Epoch 71/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3736 - acc: 0.8675
    Epoch 72/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3722 - acc: 0.8686
    Epoch 73/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.3713 - acc: 0.8688
    Epoch 74/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3698 - acc: 0.8686
    Epoch 75/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3689 - acc: 0.8695
    Epoch 76/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3674 - acc: 0.8710
    Epoch 77/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3665 - acc: 0.8697
    Epoch 78/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3655 - acc: 0.8703
    Epoch 79/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3641 - acc: 0.8704
    Epoch 80/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3630 - acc: 0.8708
    Epoch 81/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3620 - acc: 0.8716
    Epoch 82/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3613 - acc: 0.8718
    Epoch 83/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3599 - acc: 0.8724
    Epoch 84/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3593 - acc: 0.8720
    Epoch 85/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3581 - acc: 0.8726
    Epoch 86/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3566 - acc: 0.8738
    Epoch 87/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3562 - acc: 0.8736
    Epoch 88/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3550 - acc: 0.8749
    Epoch 89/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3538 - acc: 0.8739
    Epoch 90/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3532 - acc: 0.8746
    Epoch 91/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3520 - acc: 0.8751
    Epoch 92/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3511 - acc: 0.8760
    Epoch 93/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3501 - acc: 0.8762
    Epoch 94/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3493 - acc: 0.8761
    Epoch 95/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3482 - acc: 0.8770
    Epoch 96/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3472 - acc: 0.8772
    Epoch 97/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3460 - acc: 0.8773
    Epoch 98/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3455 - acc: 0.8776
    Epoch 99/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3447 - acc: 0.8785
    Epoch 100/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3438 - acc: 0.8784
    Epoch 101/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3427 - acc: 0.8788
    Epoch 102/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3421 - acc: 0.8794
    Epoch 103/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3409 - acc: 0.8796
    Epoch 104/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3400 - acc: 0.8792
    Epoch 105/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3393 - acc: 0.8803
    Epoch 106/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3380 - acc: 0.8807
    Epoch 107/120
    58500/58500 [==============================] - 2s 34us/step - loss: 0.3377 - acc: 0.8809
    Epoch 108/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3367 - acc: 0.8808
    Epoch 109/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3354 - acc: 0.8817
    Epoch 110/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3348 - acc: 0.8815
    Epoch 111/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3338 - acc: 0.8818
    Epoch 112/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3331 - acc: 0.8824
    Epoch 113/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3322 - acc: 0.8829
    Epoch 114/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3312 - acc: 0.8835
    Epoch 115/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3304 - acc: 0.8829
    Epoch 116/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3295 - acc: 0.8836
    Epoch 117/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.3289 - acc: 0.8835
    Epoch 118/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.3278 - acc: 0.8839
    Epoch 119/120
    58500/58500 [==============================] - 2s 36us/step - loss: 0.3270 - acc: 0.8839
    Epoch 120/120
    58500/58500 [==============================] - 2s 35us/step - loss: 0.3261 - acc: 0.8848
    

Recall that the dictionary `history` has two entries: the loss and the accuracy achieved using the training set.


```python
history_dict = history.history
history_dict.keys()
```




    dict_keys(['loss', 'acc'])



## Plot the results

As you might expect, we'll use our `matplotlib` for graphing. Use the data stored in the `history_dict` above to plot the loss vs epochs and the accuracy vs epochs. 


```python
# Plot the loss vs the number of epoch

key ='loss'
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(13,8))
for i in range(0,len(history_dict[key])):
    plt.scatter(i, history_dict[key][i], c='blue')
    
plt.title('Loss by Epoch')
plt.show()
```


![png](index_files/index_26_0.png)



```python
# Plot the training accuracy vs the number of epochs
key ='acc'
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(13,8))
for i in range(0,len(history_dict[key])):
    plt.scatter(i, history_dict[key][i], c='blue')
    
plt.title('Accuracy by Epoch')
plt.show()
```


![png](index_files/index_27_0.png)


It seems like we could just keep on going and accuracy would go up!

## Make predictions

Finally, it's time to make predictions. Use the relevant method discussed in the previous lesson to output (probability) predictions for the test set.


```python
# Output (probability) predictions for the test set 
y_hat_test = (model.predict(test)>.5)*1
```

## Evaluate Performance

Finally, print the loss and accuracy for both the train and test sets of the final trained model.


```python
# Print the loss and accuracy for the training set 
results_train = model.evaluate(train, label_train)
results_train
```

    58500/58500 [==============================] - 2s 36us/step
    




    [0.3193739692983974, 0.8882393162393163]




```python
# Print the loss and accuracy for the test set 
results_test = model.evaluate(test,label_test)
results_test
```

    1500/1500 [==============================] - 0s 43us/step
    




    [0.5399058806101481, 0.8039999995231628]



We can see that the training set results are really good, but the test set results lag behind. We'll talk a little more about this in the next lesson, and discuss how we can get better test set results as well!


## Additional Resources 

- https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb 
- https://catalog.data.gov/dataset/consumer-complaint-database 

## Summary 

Congratulations! In this lab, you built a neural network thanks to the tools provided by Keras! In upcoming lessons and labs we'll continue to investigate further ideas regarding how to tune and refine these models for increased accuracy and performance.
