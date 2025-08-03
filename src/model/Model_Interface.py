import torch
from arc.Binary_Architecture import Model

class ModelInterface:

    def __init__(self, model=None, model_path='', input_size=int):
        """Init method

        :param model: trained PyTorch Model.
        :param model_path: path to trained model.
        :param input_size: input size of the model.
        """
        self.model = model
        self.model_path = model_path
        self.input_size = input_size

    def load_model(self):
        """Loads a model from the specified path."""
        if self.model_path != '':
            self.model = Model(self.input_size)
            state_dict = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.set_eval()
        return self.model

    def predict(self, input_tensor):
        """Gets the model's output for the given input."""
        prediction = self.model(input_tensor)
        return prediction

    def set_eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()