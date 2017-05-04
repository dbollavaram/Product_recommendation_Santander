# Naive Bayes Third Round

## Using the new splitted dataset to train a naive bayes model. I will be using a class for the dataset, and this will help me for later using a unified class for the model. 

#Imports
import numpy as np
from sklearn.naive_bayes import BernoulliNB
import time
import gzip
from dataset import SantanderDataset #loads dataset 
from average_precision import mapk

dataset_root = './'
dataset = SantanderDataset(dataset_root)

## Testing with Naive Bayes

def train_bnb_model(msg):
    """
    Trains a model using the given parameters
    
    month: int or list with the number of the month we want
        the data to be taken of
    input_columns: a list with the name of the columns we are going to use
        in the task
    use_product: bool, if true adds the product columns of the month before
    use_change: bool, if true adds the change columns of the month before
    """
    msg_copy = msg.copy()
    msg_copy['train'] = True
    if not 'month' in msg_copy.keys():
        msg_copy['month'] = msg_copy['train_month']
    #Get the data for training
    ret = dataset.get_data(msg_copy)
    input_data, output_data = ret[0:2]
    #Fit the model
    bnb = BernoulliNB(alpha=1e-2)
    bnb.partial_fit(input_data, output_data, classes = range(24))
    return bnb


def create_prediction(bnb, msg):
    """
    Makes a prediction using the given model and parameters
    
    month: int or list with the number of the month we want
        the data to be taken of
    input_columns: a list with the name of the columns we are going to use
        in the task
    use_product: bool, if true adds the product columns of the month before
    use_change: bool, if true adds the change columns of the month before
    """
    msg_copy = msg.copy()
    msg_copy['train'] = False
    if not 'month' in msg_copy.keys():
        msg_copy['month'] = msg_copy['eval_month']
    #Get the data for making a prediction
    ret = dataset.get_data(msg_copy)
    input_data, output_data, previous_products = ret
    #Get the prediction
    rank = bnb.predict_proba(input_data)
    filtered_rank = np.equal(previous_products, 0) * rank
    predictions = np.argsort(filtered_rank, axis=1)
    predictions = predictions[:,::-1][:,0:7]
    return predictions, output_data


def naive_bayes_workflow(msg):
    """
    Implements all the steps of training and evaluating a naive bayes classifier
    Returns the score and the trained model
    
    train_month: int or list with the number of the month we want
        the data to be taken of for training 
    eval_month: int or list with the number of the month we want
        the data to be taken of for testing
    input_columns: a list with the name of the columns we are going to use
        in the task
    use_product: bool, if true adds the product columns of the month before
    use_change: bool, if true adds the change columns of the month before
    """
    if type(msg['eval_month']) is not list:
        msg['eval_month'] = [msg['eval_month']]
    #Train the model
    bnb = train_bnb_model(msg)
    scores = []
    for month in msg['eval_month']:
        msg_copy = msg.copy()
        msg_copy['month'] = month
        #Create prediction
        predictions, output_data = create_prediction(bnb, msg_copy)
        #Get the score
        score = mapk(output_data, predictions)
        scores.append(score)
    
    return scores, bnb


## Creating a submission function

def create_submission(filename, msg, 
                        verbose=False):
    """
    Implements all the steps of training and evaluating a naive bayes classifier
    Returns the score and the trained model
    
    train_month: int or list with the number of the month we want
        the data to be taken of for training 
    eval_month: int or list with the number of the month we want
        the data to be taken of for testing
    input_columns: a list with the name of the columns we are going to use
        in the task
    use_product: bool, if true adds the product columns of the month before
    use_change: bool, if true adds the change columns of the month before
    """
    test_month = 17
    #Train the model and get validation scores
    ret = naive_bayes_workflow(msg)
    scores = ret[0]
    bnb = ret[1]
    #Create a prediction
    msg['month'] = test_month
    predictions, output_data = create_prediction(bnb, msg)
    #Create the submission text
    if verbose:
        print('Creating text...')
    text='ncodpers,added_products\n'
    for i, ncodpers in enumerate(dataset.eval_current[dataset.eval_current.fecha_dato == test_month].ncodpers):
        text += '%i,' % ncodpers
        for j in predictions[i]:
            text += '%s ' % dataset.product_columns[j]
        text += '\n'
    #Write to file
    if verbose:
        print('Writing to file...')
    with gzip.open(dataset_root + 'submissions/%s.csv.gz' % filename, 'w') as f:
        f.write(bytes(text, 'utf-8'))
    
    return scores


#Create submission
start_time = time.time()
msg = {'train_month': [1,2,5,6,10,11,16],
       'eval_month': [5, 16],
      'input_columns': ['pais_residencia','age','indrel','indrel_1mes','indext','segmento','month'],
      'use_product': True,
      'use_change': True}
print(create_submission('NaiveBayes_11',msg))
print(time.time()-start_time)

## I get a LB score of , 87 in the classification( top 9%)  
## That's very good for Naive Bayes



