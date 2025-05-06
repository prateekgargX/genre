import sys
sys.path.append('../')
import os
import torch
import numpy as np
import pandas as pd

from abc import ABC
from carla.data.api import Data
from typing import List, Union, Any
from carla.models.api import MLModel
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

class IdentityMinMaxScaler(MinMaxScaler):
    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

class CustomDataCatalog(Data, ABC):
    def __init__(
            self, 
            train_y,train_X,test_y,test_X,
            data_name: str,
            YSTAR:int = 1,
            immutable_mask = None,
            cat_mask = None
            ):
        # super().__init__()

        self.ystar = YSTAR 
        # invert the labels if desired class is 0 instead of standard 1
        if YSTAR == 0:
            print("[WARNING] inverted labels used")
            train_y = 1 - train_y
            test_y  = 1 - test_y

        input_shape = train_X.shape[1]

        if immutable_mask is None: immutable_mask = [False]*input_shape
        if cat_mask is None: cat_mask = [False]*input_shape

        column_names = [f"x{i}" for i in range(input_shape)] + ['y']
        self.all_columns = column_names[:-1]
        self.catalog = {'continuous' : [column_names[i] for i in range(len(column_names[:-1])) if not(cat_mask[i])],
        'categorical' : [column_names[i] for i in range(len(column_names[:-1])) if cat_mask[i]],
        'immutable' : [column_names[i] for i in range(len(column_names[:-1])) if immutable_mask[i]],
        'target': 'y'
        }

        self.name = f"custom_{data_name}"
        self.nname = data_name
        self._df_train = pd.DataFrame(data=np.concatenate([train_X,1.0*train_y.reshape(-1,1)], axis=1),
                                      columns=column_names)
        self._df_test  = pd.DataFrame(data=np.concatenate([test_X,1.0*test_y.reshape(-1,1)], axis=1),
                                      columns=column_names)        
        self._df = pd.concat([self._df_train, self._df_test])

    @property
    def categoricals(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continous(self) -> List[str]:
        return self.catalog["continuous"]

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test.copy()

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]

    @property
    def raw(self) -> pd.DataFrame:
        return self._df_test.copy()
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        return output

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        return output


class LabelInvert(torch.nn.Module):
    # convinient wrapper to invert model
    def __init__(self,model):
        super(LabelInvert, self).__init__()
        self.baseclf = model

    def forward(self, x):
        x = self.baseclf(x)
        return 1 - x
    
class CarlaWrap(torch.nn.Module):
    """
        matches raw model to carla torch model format
    """

    def __init__(self, model):
        super(CarlaWrap,self).__init__()
        self.ann_clf = model

    def forward(self,x):
        output = torch.zeros(x.shape[0],2)
        output[:,1] = self.ann_clf(x).squeeze()
        output[:,0] = 1 - output[:,1]
        return output
    
    def half_forward(self,x):
        raise NotImplementedError("single output")
    
    def proba(self, data):
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = torch.squeeze(data)

        # might be wrong but carla implements this
        class_1 = 1 - self.forward(input)
        class_2 = self.forward(input)

        return list(zip(class_1, class_2))

    def prob_predict(self, data):
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data)

        class_1 = 1 - self.forward(input).detach().numpy().squeeze()
        class_2 = self.forward(input).detach().numpy().squeeze()

        # For single prob prediction it happens, that class_1 is casted into float after 1 - prediction
        # Additionally class_1 and class_2 have to be at least shape 1
        if not isinstance(class_1, np.ndarray):
            class_1 = np.array(class_1).reshape(1)
            class_2 = class_2.reshape(1)

        return class_2

    def predict(self, data):
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = torch.squeeze(data)
        return self.forward(input).detach().numpy()    

    
class CustomModelCatalog(MLModel):
    def __init__(
        self,
        ann_clf,
        data: CustomDataCatalog,
        model_type: str = 'ann',
        backend: str = "pytorch",
        YSTAR:int = 1
    ) -> None:
        self._model_type  = model_type
        self._backend     = backend
        self._continuous  = data.continous
        self._categorical = data.categoricals
        super().__init__(data)
                                                                                                                    
        self._feature_input_order = data.all_columns
        self._custmodel = ann_clf
        
        self._custmodel.eval()
        
        self.ystar = YSTAR
        
        if YSTAR == 0:
            print("[WARNING] model will use inverted labels")
            self._custmodel = LabelInvert(self._custmodel)
        
        self._model = CarlaWrap(self._custmodel)
        self._scaler = IdentityMinMaxScaler()

    def _test_accuracy(self):
        # get preprocessed data
        df_test = self.data.df_test
        x_test = df_test.drop(self.data.target, axis=1)
        y_test = df_test[self.data.target]

        prediction = (self.predict(x_test) > 0.5).flatten()
        correct = prediction == y_test
        print(f"test accuracy for model: {correct.mean()}")

    @property
    def feature_input_order(self) -> List[str]:
        return self._feature_input_order

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def raw_model(self) -> Any:
        return self._model

    def predict(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor]:
        if len(x.shape) != 2:
            raise ValueError(
                "Input shape has to be two-dimensional, (instances, features)."
            )

        return self.predict_proba(x)[:, 1].reshape((-1, 1))

    def predict_proba(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]:
        """
        Two-dimensional probability prediction of ml model

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction with shape N x 2
        """

        # order data (column-wise) before prediction
        # print(x, x.shape)
        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        # if self._backend == "pytorch":
        # Keep model and input on the same device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._custmodel = self._custmodel.to(device)

        if isinstance(x, pd.DataFrame):
            _x = x.values
        elif isinstance(x, torch.Tensor):
            _x = x.clone()
        else:
            _x = x.copy()

        # If the input was a tensor, return a tensor. Else return a np array.
        tensor_output = torch.is_tensor(x)
        if not tensor_output:
            _x = torch.Tensor(_x)

        _x = _x.to(device)

        output = torch.zeros(_x.shape[0],2, device=device)

        output[:,1] = self._custmodel(_x).squeeze()
        output[:,0] = 1 - output[:,1]
         
        if tensor_output:
            return output
        else:
            return output.detach().cpu().numpy()
        



