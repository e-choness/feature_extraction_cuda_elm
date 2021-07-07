import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from sklearn.decomposition import PCA
from varname import nameof
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16, DenseNet201, VGG19
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, Normalizer

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
# tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

train_101 = './Caltech101_new/train/'
test_101 = './Caltech101_new/test/'

WEIGHTS_PATH = 'E:/PycharmProjects/Pretrained Models/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        WEIGHTS_PATH,
                        cache_subdir='models')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
channels = 3
input_shape = IMAGE_SIZE + (channels,)

train_dataset = image_dataset_from_directory(train_101,
                                             shuffle=False,
                                             # label_mode='categorical',
                                             batch_size=BATCH_SIZE,
                                             image_size=IMAGE_SIZE)

test_dataset = image_dataset_from_directory(test_101,
                                            shuffle=False,
                                            # label_mode='categorical',
                                            batch_size=BATCH_SIZE,
                                            image_size=IMAGE_SIZE)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
print('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

input_tensor = Input(shape=input_shape)
x = K.applications.vgg19.preprocess_input(input_tensor)
base_mode = VGG19(include_top=False,
                  weights='imagenet',
                  input_tensor=x,
                  input_shape=input_shape
                  )
# base_mode.load_weights(weights_path)

for layer in base_mode.layers:
    layer.trainable = False
base_mode.summary()

output = base_mode.layers[-1].output
# pooling = K.layers.GlobalAveragePooling2D()
pooling = K.layers.Flatten()
output = pooling(output)

model = K.models.Model(inputs=input_tensor, outputs=output)
model.summary()

train_features = model.predict(train_dataset, batch_size=BATCH_SIZE)
test_features = model.predict(test_dataset, batch_size=BATCH_SIZE)

print('extracting features done.')

pca = PCA(n_components=255)
pca.fit(train_features)
xtrain = pca.transform(train_features)
xtest = pca.transform(test_features)

# xtrain = xtrain / np.max(xtrain)
# xtest = xtest / np.max(xtest)
scalar = Normalizer()
scalar.fit(xtrain)
xtrain = scalar.transform(xtrain)
xtest = scalar.transform(xtest)
print('features processing done.')

# add a column of ones as bias
train_bias = np.ones(shape=(xtrain.shape[0],1))
test_bias = np.ones(shape=(xtest.shape[0],1))

xtrain = np.hstack((xtrain,train_bias))
xtest = np.hstack((xtest, test_bias))

n_class = 102
ytrain = np.concatenate([y for x, y in train_dataset], axis=0)
ytrain = to_categorical(ytrain, n_class)

ytest = np.concatenate([y for x, y in test_dataset], axis=0)
ytest = to_categorical(ytest, n_class)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

# generate random weights for classifier
Win = win = np.random.uniform(0.0, 1.0, size=(256, 1000))


def save_results(x, name):
    x = x.T
    x = x.reshape(-1)
    s = './' + name + '101.csv'
    np.savetxt(s, x, delimiter=',')


name1 = nameof(xtrain)
name2 = nameof(xtest)
name3 = nameof(ytrain)
name4 = nameof(ytest)
name5 = nameof(Win)
save_results(xtrain, name1)
save_results(xtest, name2)
save_results(ytrain, name3)
save_results(ytest, name4)
save_results(Win, name5)
