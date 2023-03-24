import tensorflow as tf 
import tensorflow_hub as hub

classifier_model = "https://tfhub.dev/google/aiy/vision/classifier/food_V1/1"

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=(192, 192, 3))
])

tf.keras.models.save_model(classifier, 'model')
