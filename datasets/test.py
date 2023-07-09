import datasets

datasets = datasets.Datasets()
data, classes_data, numerical_classes, labels = datasets.read(dataset_name='student_performance')

#print first data row with labels formatted for console with each attribute in a new line
# test_data = [f'{labels[i]}: {data[0][i]}' for i in range(len(labels))]
# print('\n'.join(test_data))

#print each attribute with all possible values only once
attributes = dict()
for i in range(len(labels)):
  attributes[labels[i]] = set()
  for j in range(len(data)):
    attributes[labels[i]].add(data[j][i])
for i in attributes:
  print(f'{i}: {attributes[i]}')


