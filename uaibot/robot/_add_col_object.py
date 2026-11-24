from uaibot.utils import *


def _add_col_object(self, sim):
    if not Utils.get_uaibot_type(sim) == "uaibot.Simulation":
        raise Exception("The parameter 'sim' should be a 'uaibot.Simulation' object.")

    for i in range(len(self.links)):
        for j in range(len(self.links[i].col_objects)):
            sim.add(self.links[i].col_objects[j][0])
