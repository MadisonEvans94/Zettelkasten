#seed 
###### upstream: [[Deep Learning]], [[Software Development]], [[Python]]

[[keras]]

### Module Hierarchy: 

```md
tensorflow
│
└───keras
│   │
│   └───layers
│   │   │
│   │   └───Dense
│   │   └───Conv2D
│   │   └───MaxPooling2D
│   │   └───Dropout
│   │   └───BatchNormalization
│   │   └───Embedding
│   │   ... (many more layer types)
│   │
│   └───models
│   │   │
│   │   └───Model
│   │   └───Sequential
│   │
│   └───optimizers
│   │   │
│   │   └───SGD
│   │   └───Adam
│   │   └───RMSprop
│   │   └───Adagrad
│   │   ... (more optimizer types)
│   │
│   └───losses
│   │   │
│   │   └───BinaryCrossentropy
│   │   └───CategoricalCrossentropy
│   │   └───MeanSquaredError
│   │   └───MeanAbsoluteError
│   │   ... (more loss types)
│   │
│   └───metrics
│   │   │
│   │   └───Accuracy
│   │   └───Precision
│   │   └───Recall
│   │   └───AUC
│   │   ... (more metric types)
│   │
│   └───datasets
│   │   │
│   │   └───mnist
│   │   └───fashion_mnist
│   │   └───cifar10
│   │   └───cifar100
│   │   ... (more dataset types)
│   
└───data
│   │
│   └───Dataset
│   
└───feature_column
│   │
│   └───numeric_column
│   └───bucketized_column
│   └───categorical_column_with_vocabulary_list
│   └───embedding_column
│   └───crossed_column
│   
└───... (many more modules and classes)

```