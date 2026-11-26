def _add_col_object(self, sim):
    from uaibot.simulation import Simulation

    if not isinstance(sim, Simulation):
        raise Exception("The parameter 'sim' should be a 'uaibot.Simulation' object.")

    for i in range(len(self.links)):
        for j in range(len(self.links[i].col_objects)):
            sim.add(self.links[i].col_objects[j][0])
