import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.utils import get_balanced_sampler, get_class_weighted_cross_entropy
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

work_data = pd.read_csv(
  "/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv"
  )

numeric_variable_names = work_data.select_dtypes('number').columns.to_list()

numeric_variable_names.remove('department')

categorical_variable_names = ['department']

train, test = train_test_split(
    work_data, test_size=0.3, random_state=1001
    )  
    
train, val = train_test_split(train, random_state=1001)  

data_config = DataConfig(
    target=['separatedNY'],
    continuous_cols=numeric_variable_names,
    categorical_cols=categorical_variable_names
)

trainer_config = TrainerConfig(
    auto_lr_find=False,
    batch_size=1024,
    max_epochs=1,
    gpus=None,
    early_stopping=None
)

optimizer_config = OptimizerConfig()

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="128-64-32",
    activation="LeakyReLU", 
    learning_rate=1e-3, 
    loss='CrossEntropyLoss',
    metrics=["f1_score", "accuracy"], 
    metrics_params=[{"num_classes":2}, {}]
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

sampler = get_balanced_sampler(train['separatedNY'].values.ravel())

tabular_model.fit(train=train, validation=val, train_sampler=sampler)

result = tabular_model.evaluate(test)

tabular_model.save_model(
  "/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/tabular_dnn"
  )

loaded_model = TabularModel.load_from_checkpoint(
  "/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/tabular_dnn"
  )

result = loaded_model.evaluate(test)
