# Beans_disease_classification

Highlights of the project:

1. As the training set is too low to achieve an impressive accuracy score on the test set, heavy reliance is made on image augmentation. More specifically, tensorflow's tf.image is used. As the tf.image becomes part of the preprocessing of the images, it benefits from the asynchronous preparation of the data by the CPU. In other words, when the GPU is working on a batch, the CPU is asynchronously preparing the next batch i.e loading from the disk (unless it has not been cached), applying the necessary augmentations, shuffling it, and creating a batch from it. Moreover, with the prefetch feature, the preprocessing will happen in parallel with the training (performed by GPU).

(code for preprocessing of the training set:ds_train = ds_train.map(f).cache().shuffle(ds_info.splits['train'].num_examples).batch(BATCH_SIZE).prefetch(AUTOTUNE)

2. As overfitting happens on almost every model, I have taken some steps to combat the issue. These include image augmentaion and using drop out layers between the 2D convolution layers. 

3. I noticed the validation accuracy was fluctuating with high steps, and the model ended up with a lower validation accuracy score at the end of the last epoch. Therefore, I have the best set of weights that produce the highest validation accuracy. I achieve this by using the tf.keras.callbacks.ModelCheckpoint that keeps track of the best model at the end of every epoch.
