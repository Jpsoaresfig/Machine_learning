import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import os

# Caminho das imagens
dataset_dir = 'output'

# ImageDataGenerator com augmentação
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Caminho do modelo salvo
modelo_salvo = 'rock_paper_scissors_model.h5'

# Criação ou carregamento inicial
if os.path.exists(modelo_salvo):
    print("Carregando modelo existente...")
    model = load_model(modelo_salvo)

    # Recrie base_model para ter referência das camadas
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
else:
    print("Criando novo modelo com MobileNetV2...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_gen, epochs=5, validation_data=val_gen)
    model.save(modelo_salvo)

# Agora o fine-tuning:
for i in range(10):
    print(f'\nFine-tuning rodada {i+1}/20')

    # Liberar últimas camadas do base_model
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_gen, epochs=3, validation_data=val_gen)
    model.save(modelo_salvo)


