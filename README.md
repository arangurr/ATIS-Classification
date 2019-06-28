# Intent classification

## Repository organization

The solution is organized in the different directories and files:

- ```train.py``` trains a model with the defined architecture and saves the best performing model from the run. Execute ```python train.py -h``` for details about arguments.
- ```test.py``` test the provided model against the testing partition. Execute ```python test.py -h``` for details about arguments.
- ```utils.py``` common functions for loading the data and preprocessing.

In order to load the provided data, we use

## Data preparation

The training and testing datasets are encoded using the IOB notation. Each row constitutes one utterance, an encoded label for each token and finally the intent label. The latter is the target variable. One row could be described as follows:

Utterance | Labels
---|---
what | 0
are | 0
all | 0
the | 0
flights | 0
into | 0
atlanta's | B-toloc.city_name
airport | 0

Which has the intent label ```atis_flight``` assigned.

In order to process the data we use simple regular expressions to search for the appropriate part of the row that we are interested in:

- To extract the sentence, we extract everything in between ```BOS``` (Beginning of Sentence) and ```EOS``` (End of Sentence).
- Labels start after the  ```\t``` until it reaches the label, which always starts with ```atis_```
- The label is the rest of the line, starting from the first ```atis```

A short analysis of the sentences in and labels for the training set shows:

- There are a total of 4478 sentences
- The longest sentence has 48 tokens
- The vocabulary (number of different words) is 724 long
- There are 21 different target classes. They are distributed in the following way:

| intent | count |
---|---|
```atis_abbreviation```| 130
```atis_aircraft```| 70
```atis_aircraft#atis_flight#atis_flight_no```| 1
```atis_airfare```| 385
```atis_airline```| 139
```atis_airline#atis_flight_no```| 2
```atis_airport```| 17
```atis_capacity```| 15
```atis_cheapest```| 1
```atis_city```| 18
```atis_distance```| 17
```atis_flight```| 3309
```atis_flight#atis_airfare```| 19
```atis_flight_no```| 12
```atis_flight_time```| 45
```atis_ground_fare```| 15
```atis_ground_service```| 230
```atis_ground_service#atis_ground_fare```| 1
```atis_meal```| 6
```atis_quantity```| 41
```atis_restriction```| 5

Note that by default, some of the target classes are in reality a mix of two or more labels. For the sake of simplicity, we are not going to consider this problem as a multi-label classification, although it would be the first thing to improve if the results are not satisfactory. In any case, having classes with such different counts means we have to take the weights into acount when training a model.

## Model architecture

After the short analysis of the data, we can interpret this problem as a classification task. For our initial proposed solution, we can use a simple Recurrent Neural Netowrk. The data will be fed into an Embedding layer. Then we will choose a RNN layer (either a SimpleRNN, GRU or LSTM). Finally the classification is done in a Dense layer at the output.

## Preprocessing

In order to be able to feed the training data into our networks, we have to process it. Sentences are encoded using a tokenizer, and labels are one-hot encoded.

Any manipulation that we do on the training set has to be replicated on the testing set, so we must save a reference to the tokenizer and label encoder.

Note that we work under the assumption that the distribution of intents in the testing set is similar to that of the training set.

## Results

We check three different RNN layers: SimpleRNN, GRU and LSTM. Each network is trained for 25 epochs. The validation split is 0.2.

| Model | Best validation accuracy | No. params
| ---| --- | ---
SimpleRNN | 0.96429 | 74K
GRU |0.97210 | 107K
LSTM | 0.96652| 124K

Comparing the SimpleRNN and the GRU, we see a considerable improvement. However, it seems our model does not benefit from the additional states and parameters that the LSTM provides. Therefore, we will keep using the GRU moving forward.

A possible improvement that we can make is replace every number in the training set for ```DIG```. Intuitively, we can say that the model shouldn't need the noise from different numbers appearing in the sentences. In other words: is it important that there is a _4120_ instad of a _5311_ to decide the type of intent? Probably not, the model could only need the fact that there is a four digit number in the sentence, possibly meaning a flight number. We test this improvement on our network:

| Model | Best validation accuracy
| ---| ---
w/o replacement | 0.97210
w/ replacement |0.97433

Doing this simple trick we are able to slightly improve our metric. For the sake of brevity, we will not aim to improve the model further.

We evaluate our best model on the test set:

| Metric | Value
 ---| ---
Loss | 0.17020
Accuracy | 0.96600

## Further improvements

- Multi-label classification. As we mentioned before, we are facing this task as a multi-class classification problem. We could be more precise if we considered that a sentence could belong to several labels at the same time. This would require a bit more preprocessing and a custom labeler for the intents.
- Hyper parameter tuning. We have chosen our hyperparameters doing a simple grid search, but for a more complex model we could use the TPE algorithm. 
- Regularization measures: one of the easiest way to prevent overfitting in our model would be to add a Dropout layer.
- Data preprocessing: like we did with numbers, we could replace frequent appearances of specific types of words for a common token. This is applicable to locations, dates, currencies, etc.
- Make use of the labeled data. On the other hand, we have a significant amount of data in the labeled data that we aren't using right now. We could build a model with a double objective: slot filling and intent classification.
- Ensemble models: considering our training set is not very large, a possible way to preserve randomness during trainig would be to not fully train some  models and average their predictions.
