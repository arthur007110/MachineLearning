from typing import Literal

class Datasets:
  def __init__(self):
    self.datasets_files = dict()
    self.datasets_files['student_performance'] = 'datasets/student_performance/data.csv'
    self.datasets_files['iris'] = './iris/data.data'
    self.datasets_files['forest_fires'] = './datasets/forest_fires/forestfires.csv'

    self.datasets_configs = dict()
    self.datasets_configs['student_performance'] = {
      'labels_at_first_row': True,
      'row_separator': ';',
      'class_column': -1,
    }
    self.datasets_configs['iris'] = {
      'labels_at_first_row': False,
      'row_separator': ',',
      'class_column': -1,
    },
    self.datasets_configs['forest_fires'] = {
      'labels_at_first_row': True,
      'row_separator': ',',
      'class_column': -1,
    }

  def read(
    self,
    dataset_name: Literal["student_performance", "iris", 'forest_fires' ] = None
  ):
    if(dataset_name is None):
      return None
    dataset_file = self.datasets_files[dataset_name]
    if dataset_file is None:
      return None

    separator = self.datasets_configs[dataset_name]['row_separator']
    class_column = self.datasets_configs[dataset_name]['class_column']

    dataset = open(dataset_file, "r")
    dataset = dataset.read()
    dataset = dataset.split("\n")
    labels = []
    if(self.datasets_configs[dataset_name]['labels_at_first_row']):
      labels = dataset[0].split(separator)
      dataset = dataset[1:]
      dataset = dataset[:-1]

    data = [i.split(separator) for i in dataset]

    numerical_classes = {}
    classes_data = {}
    for i in data:
      if i[class_column] not in numerical_classes:
        numerical_classes[i[class_column]] = len(numerical_classes)
        classes_data[i[class_column]] = []
      classes_data[i[class_column]].append(i)

    return data, classes_data, numerical_classes, labels
