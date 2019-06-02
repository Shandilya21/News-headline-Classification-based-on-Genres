import numpy as np 
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.models import Model
from keras.layers import Dense,Input,Lambda
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

sess = tf.Session()


Data = open("Extracted_Cleaned.txt","r",encoding="utf8")
Data1 = pd.read_csv("Target.csv",header=None)
X=[]
for i in Data:
	X.append(i.strip())



label = LabelEncoder()
Data1["Target"] = label.fit_transform(Data1)

P=[]
for j in Data1["Target"]:
	P.append(j)

Y = np.asarray(P)
Y.reshape(Y.shape[0],1)


token = Tokenizer()
token.fit_on_texts(list(X))
X1 = token.texts_to_sequences(X)
X1 = pad_sequences(X1,maxlen=max_len,padding='post',value=0)


x_train, x_test, y_train, y_test = train_test_split(X1,Y, test_size=0.3, shuffle=True)

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
session.run(tf.global_variables_initializer())
session.run(tf.tables_initializer())
 
def ElmoEmbedding(X1):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(X1, tf.string)),"sequence_len": tf.constant(batch_size*[max_len])},signature="tokens",as_dict=True)["elmo"]

input_text = Input(shape=(max_len,), dtype=tf.string)
embedding = Lambda(ElmoEmbedding, output_shape=(None, max_len, 1024))(input_text)
dense = layers.Dense(256, activation='relu')(embedding)
pred = layers.Dense(1,activation='sigmoid')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    
/with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    history = model.fit(x_train, y_train, epochs=5, batch_size=32)
    model.save_weights('./elmo-model.h5')

with tf.Session() as session:
    K.set_session(session)

    model.load_weights('./elmo-model.h5')  
    predicts = model.predict(x_test, batch_size=32)


