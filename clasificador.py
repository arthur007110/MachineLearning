import math
filename = 'car.data'

infile = open(filename, 'r')
data = infile.read()
infile.close()

lines = data.split('\n')
cars = [line.split(',') for line in lines]

car_atributes = dict()

car_atributes['buying'] = ['vhigh', 'high', 'med', 'low']
car_atributes['maint'] = ['vhigh', 'high', 'med', 'low']
car_atributes['doors'] = ['2', '3', '4', '5more']
car_atributes['persons'] = ['2', '4', 'more']
car_atributes['lug_boot'] = ['small', 'med', 'big']
car_atributes['safety'] = ['low', 'med', 'high']

max_score = 20

car_atributes_list = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

score_classes = ['unacc', 'acc', 'good', 'vgood', 'vgood']
score_weights = [1.5, 1.5, 0.5, 0.5, 0.5, 1]

points = 0
total_cars = 1728
for car in cars:
  score = 0
  for index, atribute in enumerate(car[:-1]):
    atribute_name = car_atributes_list[index]
    score += car_atributes[atribute_name].index(atribute) * score_weights[index]

  car_class_index = math.floor(score/5.1)
  car_class = score_classes[car_class_index]
  if(car_class == car[-1]):
    points +=1
  else:
    print(car, car_class)

print(points, (points/total_cars))