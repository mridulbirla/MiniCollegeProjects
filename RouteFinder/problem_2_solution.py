# In this problem for doing BFS with segments we add each segment a weight '1' and level wise we open them where as
#  we have to do bfs with time or distance when we expand a node and when before inserting them into queue we sort them
#  in ascending order and then enqueue them. Similarly in DFS since we are you list as a stack we sort them in
#  descending order since for dfs we have LIFO operation.
# In implementing A-star algorithm when distance is the routing option then for the highways for which speed limit is not
#  given I have taken the average speed of all the highways speed limit.The heuristic is Eucledian distance between the
#  city and the end city using latitude and longitude of both city. For this we have used code from the below website
# http://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude-python.
# uses Haversine Formula for calculating it. For time based heuristic also we have used eucledian distance divided by
# speed limit.Also for city missing location we used a seperate code to get the city GPS location and populated the file
# called missing-city-gps.txt. For junctions we have used heuristic as 0 as discussed with professor

#Here we have created a class called city which has all the attributes such as connected cities their distance their
# highway name


import Queue
import sys
import math

from operator import attrgetter
from math import sin, cos, sqrt, atan2, radians


class City(object):
    def __init__(self, name, location):
        self.city_name = name
        self.connected_city_distance = {}
        self.connected_city_geo_distance = {}
        self.connected_city_time = {}
        self.connected_city_geo_time = {}
        self.connected_city_highway_name = {}
        self.connected_city_speed_limit = {}
        self.weight = 0
        self.priority = 0
        self.visited = False
        self.connected_city = []
        self.connected_city_obj = []
        self.lat = location[0]
        self.longt = location[1]
        self.path = []

    def __cmp__(self, other):
        return cmp(self.priority, other.priority)

    def update_dest(self, dest, distance, limit, highway_name):
        self.connected_city_distance[dest] = distance
        self.connected_city_time[dest] = distance / float(limit)
        self.connected_city_speed_limit[dest] = limit
        self.connected_city_highway_name[dest] = highway_name
        #  self.connected_city_geo_distance[dest] = geo_distance(self.lat, self.longt ,latitude,longitude)
        self.connected_city.append(dest)

    def update_segment_weight(self, wt):
        self.weight = 1 + wt

    def update_distance_weight(self, origin_city, wt):
        self.weight = self.connected_city_distance[origin_city] + wt

    def update_distance_heuristic_weight(self, arr_cities, origin_city, end_city, wt):

        if arr_cities[end_city].lat == '' or arr_cities[end_city].longt == '' or self.lat == '' or self.longt == '':
            t = float(wt)
        else:
            t = geo_distance(arr_cities[end_city].lat, arr_cities[end_city].longt, self.lat, self.longt)
        self.priority = self.connected_city_distance[origin_city] + t + float(wt)
        self.weight = self.connected_city_distance[origin_city] + float(wt)

    def update_time_heuristic_weight(self, arr_cities, origin_city, end_city, wt):

        if arr_cities[end_city].lat == '' or arr_cities[end_city].longt == '' or self.lat == '' or self.longt == '':
            t = float(wt)
        else:
            t = float(geo_distance(arr_cities[end_city].lat, arr_cities[end_city].longt, self.lat, self.longt)) / 47.01

        self.priority = self.connected_city_time[origin_city] + t + float(wt)
        self.weight = self.connected_city_time[origin_city] + float(wt)

    def update_time_weight(self, origin_city, wt):
        self.weight = self.connected_city_time[origin_city] + wt

    def update_city_obj(self, arr_cities):
        self.connected_city_obj = [arr_cities[i] for i in self.connected_city]

    def update_path(self, path):
        self.path = path + [self.city_name]

    def update_geo_distance(self):
        if not (self.lat == ' ' or self.longt == ' '):
            for city_obj in self.connected_city_obj:
                if city_obj.lat == ' ' or city_obj.longt == ' ':
                    self.connected_city_geo_distance[city_obj.city_name] = 0
                else:
                    self.connected_city_geo_distance[city_obj.city_name] = geo_distance(float(self.lat),
                                                                                        float(self.longt),
                                                                                        float(city_obj.lat),
                                                                                        float(city_obj.longt))
        else:
            for city_obj in self.connected_city_obj:
                self.connected_city_geo_distance[city_obj.city_name] = 0


def geo_distance(lat1, long1, lat2, long2):
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    lat1 = radians(float(lat1))
    long1 = radians(float(long1))
    lat2 = radians(float(lat2))
    long2 = radians(float(long2))

    # degrees_to_radians = math.pi / 180.0

    d_lon = long2 - long1
    d_lat = lat2 - lat1

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 3959
    distance = R * c

    return distance


def read_input():
    try:
        temp = [line.split(" ") for line in open("road-segments.txt", "r")]
        return temp
    except IOError:
        print "Missing road-segments.txt"
        sys.exit()


def read_input_location(flg):
    if flg == 1:
        fname = "city-gps.txt"
    elif flg == 2:
        fname = "missing-city-gps.txt"
    try:
        temp = [line.split(" ") for line in open(fname, "r")]

        city_dict = {}
        j = 1
        for i in temp:
            if len(i) == 1:
                continue

            if j == 5479:
                print "a"

            city_dict[i[0]] = [i[1]]
            city_dict[i[0]].append(i[2])
        return city_dict
    except IOError:
        print "Missing the file %s" % fname
        sys.exit()


def populate_cities(cities_arr, f_input, city_location):
    j = 0
    #   total_speed = 0
    for i in f_input:
        # print j
        # j += 1

        if i[3] == '':
            i[3] = 65

        if int(i[3]) == 0:
            continue

        if len(i) == 4:
            i.append(' ')
        if not i[0] in cities_arr:
            if i[0] not in city_location:
                city_location[i[0]] = ['', '']
                # print i[0]
            c = City(i[0], city_location[i[0]])
            c.update_dest(i[1], int(i[2]), int(i[3]), i[4])
            cities_arr[i[0]] = c

        else:
            t = cities_arr[i[0]]
            t.update_dest(i[1], int(i[2]), int(i[3]), i[4])
            cities_arr[i[0]] = t

        if not i[1] in cities_arr:
            if i[1] not in city_location:
                city_location[i[1]] = ['', '']
                # print i[0]
            c = City(i[1], city_location[i[1]])
            c.update_dest(i[0], int(i[2]), int(i[3]), i[4])
            cities_arr[i[1]] = c

        else:
            t = cities_arr[i[1]]
            t.update_dest(i[0], int(i[2]), int(i[3]), i[4])
            cities_arr[i[1]] = t


# total_speed += int(i[3])


def update_parent(parent, child):
    p = {}
    p[parent] = child

    return p


def execute_bfs(arr_cities, rt_option, start_city, end_city):
    bfs_q = Queue.Queue()
    bfs_q.put(arr_cities[start_city])
    parent = {}
    [arr_cities[key].update_city_obj(arr_cities) for key in arr_cities]

    arr_cities[start_city].update_path([])

    while 1:
        if bfs_q.empty():
            print "No path"
            break
        new_city = bfs_q.get()

        if new_city.city_name == end_city:
            print "%s %s" % (new_city.weight, start_city),
            print ' '.join(new_city.path)
            break

        elif new_city.visited:
            continue

        else:
            new_city.visited = True
            #            print new_city.city_name
            if rt_option == 'segments':
                [city_obj.update_segment_weight(new_city.weight) for city_obj in new_city.connected_city_obj]
            if rt_option == 'distance':
                [city_obj.update_distance_weight(new_city.city_name, new_city.weight) for city_obj in
                 new_city.connected_city_obj]
            if rt_option == 'time':
                [city_obj.update_time_weight(new_city.city_name, new_city.weight) for city_obj in
                 new_city.connected_city_obj]
                #   sorted_city = sorted(new_city.connected_city_obj, key=lambda next_city: next_city.weight)
            sorted_city = sorted(new_city.connected_city_obj, key=attrgetter('weight'), reverse=False)
            list = [bfs_q.put(s_city) for s_city in sorted_city]
            [city_obj.update_path(new_city.path) for city_obj in
             new_city.connected_city_obj]

            #   for key, value in .iteritems():
            #      parent[key] = value


def execute_dfs(arr_cities, rt_option, start_city, end_city):
    dfs_stack = []
    dfs_stack.append(arr_cities[start_city])

    [arr_cities[key].update_city_obj(arr_cities) for key in arr_cities]
    arr_cities[start_city].update_path([])
    while 1:

        new_city = dfs_stack.pop()

        if new_city.city_name == end_city:
            print "%s %s" % (new_city.weight, start_city),
            print ' '.join(new_city.path)

            break

        elif new_city.visited:
            continue

        else:
            new_city.visited = True
            #           print new_city.city_name
            if rt_option == 'segments':
                [city_obj.update_segment_weight(new_city.weight) for city_obj in new_city.connected_city_obj]
            if rt_option == 'distance':
                [city_obj.update_distance_weight(new_city.city_name, new_city.weight) for city_obj in
                 new_city.connected_city_obj]
            if rt_option == 'time':
                [city_obj.update_time_weight(new_city.city_name, new_city.weight) for city_obj in
                 new_city.connected_city_obj]
            # sorted_city = sorted(new_city.connected_city_obj, key=lambda next_city: next_city.weight)
            sorted_city = sorted(new_city.connected_city_obj, key=attrgetter('weight'), reverse=True)
            list = [dfs_stack.append(s_city) for s_city in sorted_city]
            [city_obj.update_path(new_city.path) for city_obj in new_city.connected_city_obj]


def execute_astar(arr_cities, rt_option, start_city, end_city):
    astar_q = Queue.PriorityQueue()
    astar_q.put(arr_cities[start_city])
    parent = {}
    [arr_cities[key].update_city_obj(arr_cities) for key in arr_cities]

    #  arr_cities[start_city].update_geo_distance()

    while 1:
        if astar_q.empty():
            print "No path"
            break
        new_city = astar_q.get()

        if new_city.city_name == end_city:

            print "%s %s" % (new_city.weight, start_city),
            print ' '.join(new_city.path)

            break

        elif new_city.visited:
            continue

        else:
            new_city.visited = True
            # print new_city.city_name
            if rt_option == 'segments':
                [city_obj.update_segment_weight(new_city.weight) for city_obj in new_city.connected_city_obj]
            if rt_option == 'distance':
                [city_obj.update_distance_heuristic_weight(arr_cities, new_city.city_name, end_city, new_city.weight)
                 for city_obj in
                 new_city.connected_city_obj]
            if rt_option == 'time':
                [city_obj.update_time_heuristic_weight(arr_cities, new_city.city_name, end_city, new_city.weight) for
                 city_obj in
                 new_city.connected_city_obj]
                #   sorted_city = sorted(new_city.connected_city_obj, key=lambda next_city: next_city.weight)
                # sorted_city = sorted(new_city.connected_city_obj, key=attrgetter('weight'), reverse=False)
            list = [astar_q.put(s_city) for s_city in new_city.connected_city_obj]
            [city_obj.update_path(new_city.path) for city_obj in
             new_city.connected_city_obj]

            #   for key, value in .iteritems():
            #      parent[key] = value


# Take input from User
def main():
    start_city = sys.argv[1]
    end_city = sys.argv[2]
    routing_option = sys.argv[3]
    routing_algorithm = sys.argv[4]

    # Read Input from File
    cities = {}
    temp = read_input()
    given_city_location_arr = read_input_location(1)
    missing_city_location_arr = read_input_location(2)
    city_location_arr = missing_city_location_arr.copy()
    city_location_arr.update(given_city_location_arr)
    populate_cities(cities, temp, city_location_arr)

    if start_city not in cities or end_city not in cities:
        print "Enter Valid Cities"
        sys.exit()

    if routing_algorithm == "bfs":
        execute_bfs(cities, routing_option, start_city, end_city)

    if routing_algorithm == "dfs":
        execute_dfs(cities, routing_option, start_city, end_city)

    if routing_algorithm == "astar":
        execute_astar(cities, routing_option, start_city, end_city)

main()
