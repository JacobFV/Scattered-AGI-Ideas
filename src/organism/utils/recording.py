import json
from splitstream import splitfile


def make_json_recorder(path, organ):

    file = open(f'{path}_{organ.get_name}', 'w')

    old_get_observation = organ._get_observation
    old_set_action = organ.set_action
    old_del = organ.__del__

    def new_get_observation(self):
        observation = old_get_observation()
        file.writelines(json.dumps(observation))
        return observation

    def new_set_action(self, action):
        old_set_action(action)
        file.writelines(json.dumps(action))

    def new_del(self):
        file.close()
        old_del()

    setattr(organ, 'get_observation', new_get_observation)
    setattr(organ, 'set_action', new_set_action)
    setattr(organ, '__del__', new_del)

    return organ

def make_replayer(path, organ):

    file = open(f'{path}_{organ.get_name}', 'r')
    json_iter = iter(splitfile(file, format="json"))

    old_set_action = organ.set_action
    old_del = organ.__del__

    def new_get_observation(self):
        return json.loads(next(json_iter))

    def new_set_action(self, action):
        old_set_action(json.loads(next(json_iter)))

    def new_del(self):
        del json_iter
        file.close()
        self.old_del()

    setattr(organ, 'get_observation', new_get_observation)
    setattr(organ, 'set_action', new_set_action)
    setattr(organ, '__del__', new_del)

    return organ


def make_byte_object_recorder(path, organ):
    raise NotImplementedError()