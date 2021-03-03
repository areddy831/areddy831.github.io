from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet')
model.save('models/model_vgg16_imagenet.h5')
print('Pre-trained VGG16 model with ImageNet weights saved!')