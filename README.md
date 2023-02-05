# Beans_disease_classification
As the training set is too low to achieve an impressive accuracy score on the test set, heavy reliance is made on image augmentation. More specifically, tensorflow's tf.image is used. As the tf.image becomes part of the preprocessing of the images, it benefits from the asynchronous preparation of the data by the CPU. In other words, when the GPU is working on a batch, the CPU is asynchronously preparing the next batch i.e loading from the disk (unless it has not been cached), applying the necessary augmentations, shuffling it, and creating a batch from it. Moreover, with the prefetch feature, the preprocessing will happen in parallel with the training (performed by GPU).

(code for preprocessing of the training set:ds_train = ds_train.map(f).cache().shuffle(ds_info.splits['train'].num_examples).batch(BATCH_SIZE).prefetch(AUTOTUNE)
