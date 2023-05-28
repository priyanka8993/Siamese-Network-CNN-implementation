from imutils import build_montages
import numpy as np
import pandas as pd

##Sample index combination and splitting
from sklearn.model_selection import  StratifiedShuffleSplit
from itertools import combinations_with_replacement

##Packages to process images
from PIL import Image
from PIL import UnidentifiedImageError
import cv2

## Packages to load file from s3
import boto3
import s3fs
import re


## Packages to build siamese network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

AWS_KEY_ID ="AKIAIIFXPX54K2E4RU6A"
AWS_SECRET_ACCESS = "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"

def get_image_list(s3_full_path):
    print("Creating list of image path..")
    fs = s3fs.S3FileSystem()
    saree_image_list = fs.glob(s3_full_path)
    return saree_image_list

def read_images(image_list, bucket_name):
    print("Reading images into array...")
    image_array_list = []
    image_id_list = []
    s3 = boto3.resource('s3', 
                        region_name='ap-southeast-1', 
                        aws_access_key_id=AWS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS)
    
    
    bucket = s3.Bucket(bucket_name)
    
    for current_image in image_list:
        try:
            image_name = re.sub(bucket_name+'/', '',current_image)
            image_id = re.sub(".jpg", "",image_name.split("/")[-1])
            image_id_list.append(image_id)
            object = bucket.Object(image_name)
            response = object.get()
            file_stream = response['Body']
            im = np.array(Image.open(file_stream))
            image_array_list.append(im)
        except UnidentifiedImageError:
            print("Exception")      
    return image_array_list,image_id_list


def make_pairs(images, labels):
    print ("Creating image pairs .....")
   
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    imageIdPairList = list(combinations_with_replacement(labels,2))
    n = 0
    for x,y in combinations_with_replacement(images,2):   
        imageA = cv2.resize(x, (100,100), interpolation=cv2.INTER_LINEAR)
        imageB = cv2.resize(y, (100,100), interpolation=cv2.INTER_LINEAR)
        pairImages.append([imageA,imageB])
        ImageIDPair = imageIdPairList[n]
        n = n + 1
        if ImageIDPair[0] == ImageIDPair[1]:
            pairLabels.append(1)
        elif ImageIDPair[0] in ImageIDPair[1]:
            pairLabels.append(1)
        elif ImageIDPair[1] in ImageIDPair[0]:
            pairLabels.append(1)
        else:
            pairLabels.append(0)
            
    pairImagesArray = np.array(pairImages)
    pairLabelsArray = np.array(pairLabels)
    
    return pairImagesArray,pairLabelsArray


def make_train_test_split(pairAll, labelAll):
    
    print("Create test, train, validation set")
    print(pairAll[1:2])
    print(labelAll[1:2])
    ## Create train, test, and validation set
    ss = StratifiedShuffleSplit(1, train_size=0.8)

    ## Train and test split
    train_ix, test_ix = next(ss.split(pairAll,labelAll))
    pairTrain, pairTest = pairAll[train_ix], pairAll[test_ix]
    labelTrain, labelTest = labelAll[train_ix], labelAll[test_ix]

    ## Test and validation split
    newTest_ix, Val_ix =next(ss.split(pairTest,labelTest))
    pairTestNew, pairVal = pairTest[newTest_ix], pairTest[Val_ix]
    labelTestNew, labelVal = labelTest[newTest_ix], labelTest[Val_ix]
    
    return([pairTrain,labelTrain],[pairTestNew,labelTestNew],[pairVal,labelVal])




def build_siamese_model(inputShape, embeddingDim=48):
    print("Building the siamese model...")
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(32, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Dense(72, 
              activation='relu',
              name = 'Embeddings')(x)
    
    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)
    
    # build the model
    model = Model(inputs, outputs)
    
    # return the model to the calling function
    return model

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def build_network(pairTrain, labelTrain, pairTestNew, labelTestNew):
    # specify the shape of the inputs for our network
    IMG_SHAPE = (100, 100, 3)
    # specify the batch size and number of epochs
    BATCH_SIZE = 64
    EPOCHS = 100
    
    imgA = Input(shape=IMG_SHAPE)
    imgB = Input(shape=IMG_SHAPE)
    featureExtractor = build_siamese_model(IMG_SHAPE)
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)
    
    # finally, construct the siamese network
    distance = Lambda(euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs)
    
    # compile the model
    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer="adam",
    metrics=["accuracy"])
    
    # train the model
    print("[INFO] training model...")
    history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
    validation_data=([pairTestNew[:, 0], pairTestNew[:, 1]], labelTestNew[:]),
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS)
        
    return model


def train_siamese_network(input_s3_path,s3_bucket):
    
    ## Get image lists
    all_image_s3_path = get_image_list(input_s3_path)
    
    ##Create image array from image list and get corresponding ids 
    all_image_array, all_image_ids = read_images(all_image_s3_path, s3_bucket)
    
    ##Create image pairs from list of all images
    all_image_pairs,all_pair_labels = make_pairs(all_image_array,all_image_ids)
    
    ## Create train, test, and validation set
    TrainSet, TestSet, ValSet = make_train_test_split(all_image_pairs,all_pair_labels)
    
    ## Build the model
    siameseModel = build_network(TrainSet[0],TrainSet[1],TestSet[0],TestSet[1])
    
    ## Test model on the validation set
    print("Testing the trained model on validation set...")
    pairVal = ValSet[0]
    labelVal = ValSet[1]
    pred_val = siameseModel.predict([pairVal[:,0],pairVal[:,1]])
    
    predictions_df = pd.DataFrame({"True_Value" : labelVal, "Pred_Value" : pred_val})
    
    ##Write validation prediction accuracy in a file (## save to s3 bucket)
    prediction_file_name = input_s3_path+"_predictions.csv"
    session = boto3.Session(aws_access_key_id=AWS_KEY_ID,
                            aws_secret_access_key=AWS_SECRET_ACCESS)
    s3 = session.resource('s3')
    csv_buffer = StringIO()
    predictions_df.to_csv(pred_csv_buffer)
    object = s3.Object('reshamandi-ml-data',prediction_file_name)
    object.put(Body=pred_csv_buffer.getvalue())
    
    
    return siameseModel
    
# if __name__ == '__main__':
        
#     # Data, model, and output directories
#     parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
#     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
#     parser.add_argument('--train-data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_DATA'))
#     parser.add_argument('--bucket', type=str, default=os.environ.get('SM_CHANNEL_BUCKET'))

#     args, _ = parser.parse_known_args()
    
#     train_data = args.train_data
#     bucket = args.bucket
#     prefix = args.prefix
#     model_dir  = args.model_dir
#     export_dir=os.path.join(os.environ.get('SM_MODEL_DIR'), 'export', 'Servo', '1')
    
#     tf.saved_model.save(model, export_dir)
