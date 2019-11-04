# coding = utf8
# coding by liuyunfei


import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras import Model

tf.keras.backend.set_floatx('float32')
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

x_train = x_train[...,tf.newaxis]
x_test = x_test[...,tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1 = Conv2D(32,3,activation='relu')        
        self.conv2 = Conv2D(64,3,activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128,activation='relu')
        self.d2 = Dense(10,activation='softmax')

    def call(self,x):
        x = self.conv1(x)
        x = self.conv2(x)        
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

# 选择优化器和损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')

@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels,predictions)
    gradients =tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    train_loss(loss)
    train_acc(labels,predictions)

@tf.function
def test_step(images,labels):
    
    predictions = model(images)
    t_loss = loss_object(labels,predictions)

    train_loss(t_loss)
    train_acc(labels,predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    for images,labels in train_ds:
        train_step(images,labels)
    for test_images,test_labels in test_ds:
        test_step(test_images,test_labels)
    
    template = 'Epoch {}, Loss: {}, Acc: {} , TestLoss:{},TestAcc:{}'
    print(template.format(
        epoch+1,
        train_loss.result(),
        train_acc.result()*100,
        test_loss.result(),
        test_acc.result()*100
    ))
















