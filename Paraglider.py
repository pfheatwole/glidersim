class Paraglider:
    def __init__(self, wing, S_cg, Cd_cg):
        """
        Parameters
        ----------
        wing : ParagliderWing
        S_cg : float [meters**2]
            Surface area of the cg
        Cd_cg : float [N/m**2]
            Drag coefficient of cg
        """
        self.wing = wing
        self.S_cg = S_cg  # FIXME: move into a Harness?
        self.Cd_cg = Cd_cg  # FIXME: move into a Harness?
