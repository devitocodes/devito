from examples.seismic import SeismicModel

class ISOSeismicModel(SeismicModel):

    def _initialize_physics(self, vp, space_order, **kwargs):
        print("MODEL NOVO")
        params = []
        # Buoyancy
        b = kwargs.get('b', 1)

        self.rho = self._gen_phys_param(1/b, 'rho', space_order)

        # Initialize elastic with Lame parametrization
        vs = kwargs.pop('vs')
        self.lam = self._gen_phys_param((vp**2 - 2. * vs**2)/b, 'lam', space_order,
                                        is_param=True)
        self.mu = self._gen_phys_param(vs**2 / b, 'mu', space_order, is_param=True)
        self.vs = self._gen_phys_param(vs, 'vs', space_order)
        self.vp = self._gen_phys_param(vp, 'vp', space_order)
        
        # Initialize rest of the input physical parameters
        for name in self._known_parameters:
            if kwargs.get(name) is not None:
                field = self._gen_phys_param(kwargs.get(name), name, space_order)
                setattr(self, name, field)
                params.append(name)
