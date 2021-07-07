import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from varname import nameof
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16, DenseNet201, VGG19, InceptionResNetV2, ResNet152V2, MobileNetV3Large
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
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        # tf.config.experimental.set_virtual_device_configuration(gpus[0],
        # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

train_256 = './Caltech256_new/train/'
test_256 = './Caltech256_new/test/'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
channels = 3
input_shape = IMAGE_SIZE + (channels,)

train_dataset = image_dataset_from_directory(train_256,
                                             shuffle=False,
                                             # label_mode='categorical',
                                             batch_size=BATCH_SIZE,
                                             image_size=IMAGE_SIZE)

test_dataset = image_dataset_from_directory(test_256,
                                            shuffle=False,
                                            # label_mode='categorical',
                                            batch_size=BATCH_SIZE,
                                            image_size=IMAGE_SIZE)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
print('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


def vgg16_places(train_dataset, test_dataset, last_layer=-1):
    input_tensor = Input(shape=input_shape)
    # WEIGHTS_PATH = 'E:/PycharmProjects/Pretrained Models/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                         WEIGHTS_PATH,
    #                         cache_subdir='models')
    WEIGHTS_PATH = 'C:/Users/Admin/.keras/models/vgg16-hybrid1365_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16-hybrid1365_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH,
                            cache_subdir='models')

    x = K.applications.vgg16.preprocess_input(input_tensor)
    base_mode = VGG16(include_top=False,
                      # weights='imagenet',
                      weights=weights_path,
                      input_tensor=x,
                      input_shape=input_shape,
                      classes=1365)
    # base_mode.load_weights(weights_path)

    for layer in base_mode.layers:
        layer.trainable = False
    # base_mode.summary()

    for layer in base_mode.layers:
        layer.trainable = False
    base_mode.summary()

    output = base_mode.layers[last_layer].output
    pooling = K.layers.GlobalAveragePooling2D()
    # pooling = K.layers.Flatten()
    output = pooling(output)

    model = K.models.Model(inputs=input_tensor, outputs=output)
    model.summary()

    train_features = model.predict(train_dataset, batch_size=BATCH_SIZE)
    test_features = model.predict(test_dataset, batch_size=BATCH_SIZE)

    return train_features, test_features


def models_lastlayer(train_dataset, test_dataset, model_name, preprocess, last_layer=-1):
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    channels = 3
    input_shape = IMAGE_SIZE + (channels,)
    input_tensor = Input(shape=input_shape)

    if model_name == 'dense201':
        b_model = DenseNet201
    if model_name == 'res152':
        b_model = ResNet152V2
    if model_name == 'inception':
        b_model = InceptionResNetV2
    if model_name == 'mobile3':
        b_model = MobileNetV3Large

    x = preprocess(input_tensor)
    base_mode = b_model(include_top=False,
                        weights='imagenet',
                        # weights=weights_path,
                        input_tensor=x,
                        input_shape=input_shape,
                        classes=1000)
    # base_mode.load_weights(weights_path)

    for layer in base_mode.layers:
        layer.trainable = False
    # base_mode.summary()

    output = base_mode.layers[last_layer].output
    pooling = K.layers.GlobalAveragePooling2D()
    # pooling = K.layers.Flatten()
    output = pooling(output)

    model = K.models.Model(inputs=input_tensor, outputs=output)
    # model.summary()
    print(model.output_shape)

    train_features = model.predict(train_dataset, batch_size=BATCH_SIZE)
    test_features = model.predict(test_dataset, batch_size=BATCH_SIZE)

    return train_features, test_features


def run_encoder(train_features, test_features, n_feature=256):
    regularizer = K.regularizers.l1(0.000001)

    print(train_features.shape[1])

    auto_input = Input(shape=(train_features.shape[1],))
    # encoded = Dense(1024,
    #                 activation='relu',
    #                 activity_regularizer=regularizer
    #                 )(auto_input)
    decoded_input = Dense(n_feature,
                          activation='relu',
                          activity_regularizer=regularizer
                          )(auto_input)
    # decoded = Dense(1024, activation='relu')(decoded_input)
    decoded = Dense(train_features.shape[1], activation='sigmoid')(decoded_input)

    autoencoder = Model(inputs=auto_input, outputs=decoded)
    encoder = Model(inputs=auto_input, outputs=decoded_input)
    optimizer = K.optimizers.deserialize(0.0001)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(train_features, train_features, epochs=50, batch_size=32,
                    shuffle=True, validation_data=(test_features, test_features))

    train_encoded = encoder.predict(train_features)
    test_encoded = encoder.predict(test_features)
    print('Third feature extraction done.')
    return train_encoded, test_encoded


def run_pca(train_data, test_data, n_component=511):
    pca = PCA(n_components=n_component)
    pca.fit(train_data)
    xtrain = pca.transform(train_data)
    xtest = pca.transform(test_data)
    print('PCA done.')
    return xtrain, xtest


def normalization(xtrain, xtest):
    scalar = Normalizer()
    scalar.fit(xtrain)
    xtrain = scalar.transform(xtrain)
    xtest = scalar.transform(xtest)
    print('Normalizing done.')
    return xtrain, xtest


preprocess1 = K.applications.densenet.preprocess_input


train_feature1, test_feature1 = models_lastlayer(train_dataset, test_dataset, 'dense201', preprocess1, last_layer=-1)
train_feature2, test_feature2 = models_lastlayer(train_dataset, test_dataset, 'dense201', preprocess1, last_layer=-5)
train_feature3, test_feature3 = models_lastlayer(train_dataset, test_dataset, 'dense201', preprocess1, last_layer=-8)#vgg16_places(train_dataset, test_dataset, last_layer=-1)

# train_encoded1, test_encoded1 = run_encoder(train_feature1, test_feature1, n_feature=256)
# train_encoded2, test_encoded2 = run_encoder(train_feature2, test_feature2, n_feature=32)
# train_encoded3, test_encoded3 = run_encoder(train_feature3, test_feature3, n_feature=512)

train_encoded = np.hstack((train_feature1, train_feature2, train_feature3))
test_encoded = np.hstack((test_feature1, test_feature2, test_feature3))

print(train_encoded.shape)
print(test_encoded.shape)

feature_number = 511
xtrain, xtest = train_encoded, test_encoded
xtrain, xtest = run_pca(train_encoded, test_encoded, feature_number)
# xtrain, xtest = run_encoder(train_feature, test_feature)
xtrain, xtest = normalization(xtrain, xtest)

Win = win = np.random.uniform(0.0, 1.0, size=(xtrain.shape[1] + 1, 1000))

print(Win.shape)
#

# add a column of ones as bias
train_bias = np.ones(shape=(xtrain.shape[0], 1))
test_bias = np.ones(shape=(xtest.shape[0], 1))

xtrain = np.hstack((xtrain, train_bias))
xtest = np.hstack((xtest, test_bias))

# xtrain = xtrain / np.max(xtrain)
# xtest = xtest / np.max(xtest)
# scalar = Normalizer()
# scalar.fit(xtrain)
# xtrain = scalar.transform(xtrain)
# xtest = scalar.transform(xtest)
# print('features processing done.')

n_class = 257
ytrain = np.concatenate([y for x, y in train_dataset], axis=0)
ytrain = to_categorical(ytrain, n_class)

ytest = np.concatenate([y for x, y in test_dataset], axis=0)
ytest = to_categorical(ytest, n_class)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# preprocessing data for cuda, since cublas functions are column major
def save_results(x, name):
    x = x.T
    x = x.reshape(-1)
    s = './' + name + '256.csv'
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
# np.savetxt("./xtrain1.csv", xtrain, delimiter=",")
# np.savetxt("./xtest1.csv", xtest, delimiter=",")
# np.savetxt("./ytrain1.csv", ytrain, delimiter=",")
# np.savetxt("./ytest1.csv", ytest, delimiter=",")
