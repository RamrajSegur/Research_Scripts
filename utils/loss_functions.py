from keras import backend as K
import tensorflow as tf


def weighted_binary_crossentropy():
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        weights=tf.count_nonzero(y_true,axis=1,keep_dims=True)
#         weights=tf.count_nonzero(y_true,axis=1)
        weights=tf.to_float(weights)
        loss = (y_true * K.log(y_pred)+(1-y_true)*K.log(1-y_pred))*[1,3,6]*(1-(weights/tf.to_float(tf.size(y_true)/3)))
#         loss = (y_true * K.log(y_pred)+(1-y_true)*K.log(1-y_pred))*(1/tf.to_float(tf.size(y_true)))
        loss = -K.sum(loss, -1)
        return loss
    return loss

def weighted_pixelwise_crossentropy(class_weights):

    def loss(y_true, y_pred):
        epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))

    return loss


def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)
