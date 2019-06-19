from keras.models import Model
from keras.layers import Dense, Input, Activation, multiply, Lambda
from keras.layers import TimeDistributed, GRU, Bidirectional
from keras import backend as K
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


def han(input_dim=(11, 40, 200), output_dim=3):
    # refer to 4.2 in the paper whil reading the following code
    C, W, H = input_dim

    # Input for one day : max article per day =40, dim_vec=200
    input1 = Input(shape=(W, H), dtype='float32')

    # Attention Layer
    dense_layer = Dense(200, activation='tanh')(input1)
    softmax_layer = Activation('softmax')(dense_layer)
    attention_mul = multiply([softmax_layer,input1])
    #end attention layer
    
    
    vec_sum = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)
    pre_model1 = Model(input1, vec_sum)
    pre_model2 = Model(input1, vec_sum)

    # Input of the HAN shape (None,11,0,200)
    # 11 = Window size = N in the paper 40 = max articles per day, dim_vec = 200
    input2 = Input(shape=(C, W, H), dtype='float32')

    # TimeDistributed is used to apply a layer to every temporal slice of an input 
    # So we use it here to apply our attention layer ( pre_model ) to every article in one day
    # to focus on the most critical article
    pre_gru = TimeDistributed(pre_model1)(input2)

    # bidirectional gru
    l_gru = Bidirectional(GRU(100, return_sequences=True))(pre_gru)
    
    # We apply attention layer to every day to focus on the most critical day    
    post_gru = TimeDistributed(pre_model2)(l_gru)

    # MLP to perform classification
    dense1 = Dense(100, activation='tanh')(post_gru)
    dense2 = Dense(output_dim, activation='tanh')(dense1)
    final = Activation('softmax')(dense2)
    final_model = Model(input2, final)
    #final_model.summary()

    return final_model
