extract_class = 'good'

filename = 'car.data'

infile = open(filename, 'r')
data = infile.read()
infile.close()

lines = data.split('\n')
cars = [line.split(',') for line in lines]

filtered_cars = []

for car in cars:
  if(car[-1] == extract_class):
    filtered_cars.append(car)

outfile = open(f'{extract_class}.data', 'w')
for car in filtered_cars:
  outfile.write(',\t\t\t'.join(car) + '\n')