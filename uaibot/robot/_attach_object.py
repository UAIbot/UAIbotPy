from uaibot.utils import *


def _attach_object(self, obj):
    # Error handling

    if not (Utils.get_uaibot_type(obj) in Utils.IS_GROUPABLE):
        raise Exception("The parameter 'obj' must be one of the following types: " + str(Utils.IS_GROUPABLE) + ".")
    # end error handling

    self._attached_objects.append([obj, Utils.inv_htm(self.fkm()) @ obj.htm])
