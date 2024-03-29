from keras.applications import vgg19
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

img_rows, img_cols=224,224

model=vgg19.VGG19(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))

model.trainable=False


def layer_adder(bottom_model,num_classes):
    top_model=bottom_model.output
    top_model=GlobalAveragePooling2D()(top_model)
    top_model=Dense(1024,activation='relu')(top_model)
    top_model=Dense(1024,activation='relu')(top_model)
    top_model=Dense(512,activation='relu')(top_model)
    top_model=Dense(num_classes,activation='softmax')(top_model)
    return top_model



num_classes=2
FC_head=layer_adder(model,num_classes)

model=Model(model.input,outputs=FC_head)

print(model.summary())



#importing and preprocessing data to be fed to the neural network

from keras.preprocessing.image import ImageDataGenerator

train_data_dir=r'train'
validation_data_dir=r'test'


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')
validation_datagen=ImageDataGenerator(rescale=1./255)

batch_size=1

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows,img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator= validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows,img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)



from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint= ModelCheckpoint(
    "face_recognition.h5",
    monitor="val_loss",
    mode="min",
    save_best_only= True,
    verbose=1
)

earlystop=EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=1,
    restore_best_weights=True
)

callbacks=[earlystop,checkpoint]

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy'])

nb_train_samples=1200
nb_validation_samples=52

epochs=10
batch_size=16

history=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples
)