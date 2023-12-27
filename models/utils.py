import random, itertools
import numpy as np

def locate_instances(num_instances, minimas, total_area, im_map, normalized = True):
    WIDTH = 39
    # Check if num_instances == num_minimas, if its the case we have found our peoples
    if num_instances == len(minimas):
        peoples = [[min['x'] + min['w']//2, min['y'] + min['h']//2] for min in minimas]
        if normalized:
            peoples = [[people[0] / WIDTH, people[1] / WIDTH] for people in peoples]
        return peoples
    # Compute amplitudes
    for min in minimas:
        min['ampl'] = (min['h'] * min['w']) #/ total_area
    minimas.sort(key=lambda min: min['ampl'])
    if num_instances < len(minimas):
        peoples = [[min['x'] + min['w']//2, min['y'] + min['h']//2] for min in minimas[:-num_instances.int().item()]]
        if normalized:
            peoples = [[people[0] / WIDTH, people[1] / WIDTH] for people in peoples]
        return peoples
    else:
        # print(f"Num_instances: {num_instances} | len(minimas) {len(minimas)}")
        peoples = []
        # lets estimate a person's size
        instance_size = total_area / num_instances
        # minimas.reverse()
        for id, min in enumerate(minimas):
            num_peoples = round(min['ampl'] / instance_size.item())
            # print(f"num people for min {id} | num_peoples {num_peoples} | area {min['ampl']} | instance_size {instance_size} | total_area {total_area}")
            if num_peoples == 0 or num_peoples == 1:
                x = min['x'] + min['w']//2
                y = min['y'] + min['h']//2
                peoples += [[x,y]]
            else:
                region = im_map[min['y']: min['y'] + min['h'], min['x']: min['x'] + min['w']]
                possible_coordinates = list(zip(*np.where(region == 1)))
                possible_coordinates = [(pos[0] + min['x'], pos[1] + min['y']) for pos in possible_coordinates]
                indices = random.choices(range(len(possible_coordinates)), k=num_peoples)
                peoples += [possible_coordinates[i] for i in indices]
    if num_instances < len(peoples):
        peoples = peoples[:num_instances.int().item()]

    if normalized:
        peoples = [[people[0] / WIDTH, people[1] / WIDTH] for people in peoples]

    return peoples