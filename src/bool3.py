class B:
    # Define the three possible values
    TRUE = 2
    FALSE = 0
    UNASSIGNED = 1

    def __init__(self, value=None):
        if value is None:
            self.value = B.UNASSIGNED
        elif value in (True, False):
            self.value = B.TRUE if value else B.FALSE
        elif value in (B.TRUE, B.FALSE, B.UNASSIGNED):
            self.value = value
        else:
            raise ValueError(f"Invalid value for B: {value}")

    def __repr__(self):
        if self.value == B.TRUE:
            return "True"
        elif self.value == B.FALSE:
            return "False"
        elif self.value == B.UNASSIGNED:
            return "Unassigned"

    def __eq__(self, other):
        if isinstance(other, B):
            return self.value == other.value
        return False

    def __and__(self, other):
        if isinstance(other, B):
            if self.value == B.FALSE or other.value == B.FALSE:
                return B(B.FALSE)
            elif self.value == B.UNASSIGNED or other.value == B.UNASSIGNED:
                return B(B.UNASSIGNED)
            else:
                return B(B.TRUE)
        raise ValueError("Can only perform logical operations between B instances")

    def __or__(self, other):
        if isinstance(other, B):
            if self.value == B.TRUE or other.value == B.TRUE:
                return B(B.TRUE)
            elif self.value == B.UNASSIGNED or other.value == B.UNASSIGNED:
                return B(B.UNASSIGNED)
            else:
                return B(B.FALSE)
        raise ValueError("Can only perform logical operations between B instances")

    def __invert__(self):
        if self.value == B.TRUE:
            return B(B.FALSE)
        elif self.value == B.FALSE:
            return B(B.TRUE)
        elif self.value == B.UNASSIGNED:
            return B(B.UNASSIGNED)

    def implies(self, other):
        """Custom method for Implies operation (A → B)."""
        if isinstance(other, B):
            if self.value == B.FALSE:
                return B(B.TRUE)  # False → Anything = True
            elif self.value == B.TRUE:
                return other  # True → B = B
            elif self.value == B.UNASSIGNED:
                if other.value == B.TRUE:
                    return B(B.TRUE)  # Unassigned → True = True
                else:
                    return B(B.UNASSIGNED)
        raise ValueError("Can only perform logical operations between B instances")

    def __rshift__(self, other):
        return self.implies(other)

    def __xor__(self, other):
        if isinstance(other, B):
            if self.value == B.UNASSIGNED or other.value == B.UNASSIGNED:
                return B(B.UNASSIGNED)
            elif self.value == other.value:
                return B(B.FALSE)  # Same values = False
            else:
                return B(B.TRUE)  # Different values = True
        raise ValueError("Can only perform logical operations between B instances")

