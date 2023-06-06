
def rateCar(car):
  buying, maint, doors, persons, lug_boot, safety, rating = car

  if(safety == 'low'):
    return 'unacc'
  if(persons == '2'):
    return 'unacc'
  
  if(maint == 'vhigh'):
    return 'unacc'
  
  if(buying in ['high', 'vhigh']):
    return 'acc'

  if(lug_boot != 'small' and safety == 'high'):
    return 'vgood'

  if(maint in ['low', 'med'] ):
    return 'good'
    
  return 'acc'

def printCar(car, end='[]\n'):
  print(' \t'.join(car), end=end)


filename = 'car.data'

infile = open(filename, 'r')
data = infile.read()
infile.close()

lines = data.split('\n')
cars = [line.split(',') for line in lines]



max_score = 20

car_atributes_list = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

score_classes = ['unacc', 'acc', 'good', 'vgood', 'vgood']

classified_cars = dict()
total_classified_cars = {
  'unacc': 1210,
  'acc': 384,
  'good': 69,
  'vgood': 65
}

correct_classified_cars = {
  'unacc': 0,
  'acc': 0,
  'good': 0,
  'vgood': 0
}

correct = 0
total_cars = len(cars)
for car in cars:
  car_class = rateCar(car)

  if(car_class not in classified_cars):
    classified_cars[car_class] = 1
  else:
    classified_cars[car_class] += 1
  
  if(car_class == car[-1]):
    correct_classified_cars[car_class] += 1
    correct += 1

print('Classified cars:')
for car_class in classified_cars:
  print(f'{car_class}: {classified_cars[car_class]}')
  print(f'Corectly classified: {correct_classified_cars[car_class]}')
  print(f'Total from class: {total_classified_cars[car_class]}')
  print(f'Accuracy: {round(correct_classified_cars[car_class] / total_classified_cars[car_class] * 100, 2)}%')
  print('----------------------------------------------')
print(f'Total Accuracy: {round(correct / total_cars * 100, 2)}%')
