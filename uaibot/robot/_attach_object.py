from uaibot.utils import is_a_groupable_object, IS_GROUPABLE, inv_htm


def _attach_object(self, obj):
    # Error handling

    if not (is_a_groupable_object(obj)):
        raise Exception(
            "The parameter 'obj' must be one of the following types: "
            + str(IS_GROUPABLE)
            + "."
        )
    if obj is self:
        raise ValueError("A manipulator cannot be attached to itself.")
    # end error handling

    self._attached_objects.append([obj, inv_htm(self.fkm()) @ obj.htm])
