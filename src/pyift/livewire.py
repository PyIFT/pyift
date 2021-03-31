import numpy as np
import _pyift
import collections
from typing import Union, Sequence, Optional
import warnings


class LiveWire:
    current: Optional[np.ndarray]
    saliency: Optional[np.ndarray]

    def __init__(self, image: np.ndarray, arc_fun: str = 'exp', saliency: Optional[np.ndarray] = None, **kwargs):
        """
        Live-wire object to iteratively compute the optimum-paths [1]_ between user selected points.

        Parameters
        ----------
        image: array_like
            Array where the first two dimensions are the image domain, the third and optional are its features.
        arc_fun: {'exp'}, default='exp'
            Optimum-path arc-weight function.
        saliency: array_like, optional
            Array with the same dimension as the image domain containing the foreground saliency.
        kwargs: float, optional
            Key word arguments for arc-weight function parameters.

        Attributes
        ----------
        arc_fun: {'exp'}, default='exp'
            Optimum-path arc-weight function.
        image: array_like
            Array where the first two dimensions are the image domain, the third and optional are its features.
        saliency: array_like, optional
            Array with additional features, usually object saliency. Must have the same domain as `image`.
        costs: array_like
            Array containing the optimum-path cost.
        preds: array_like
            Array containing the predecessor map to recover the optimal contour.
        labels: array_like
            Array indicating optimum-path nodes
        size: tuple
            Tuple containing the image domain.
        sigma: float
            Image features parameter.
        gamma: float
            Saliency features parameter.
        source: int
            Current path source node index (flattened array), -1 if inactive.
        destiny: int
            Current path destiny node index (flattened array), -1 if inactive.
        start: int
            Current contour starting node index, -1 if inactive.
        current: array_like
            Active optimum-path, before confirmation.
        paths: dict
            Ordered dictionary of paths, key: path source, value: path sequence.

        Examples
        --------

        >>> import numpy as np
        >>> from pyift.livewire import LiveWire
        >>>
        >>> image = np.array([[8, 1, 0, 2, 0],
        >>>                   [5, 7, 2, 0, 1],
        >>>                   [6, 7, 6, 1, 0],
        >>>                   [6, 8, 7, 0, 3],
        >>>                   [6, 7, 8, 8, 9]])
        >>>
        >>> lw = LiveWire(image, sigma=1.0)
        >>> lw.select((0, 0))
        >>> lw.confirm()
        >>> lw.select((4, 4))
        >>> lw.confirm()
        >>> lw.contour

        References
        ----------
        .. [1] Falcão, Alexandre X., et al. "User-steered image segmentation paradigms:
               Live wire and live lane." Graphical models and image processing 60.4 (1998): 233-260.

        """
        if not isinstance(image, np.ndarray):
            raise TypeError('`image` must be a `ndarray`.')

        if image.ndim == 2:
            image = np.expand_dims(image, 2)

        if image.ndim != 3:
            raise ValueError('`image` must 2 or 3-dimensional array.')

        if saliency is not None:
            if not isinstance(saliency, np.ndarray):
                raise TypeError('`saliency` must be a `ndarray`.')

            if saliency.ndim == 2:
                saliency = np.expand_dims(saliency, 2)

            if saliency.ndim != 3:
                raise ValueError('`saliency` must 2 or 3-dimensional array.')

            if saliency.shape[:2] != image.shape[:2]:
                raise ValueError('`saliency` and `image` 0,1-dimensions must match.')

            self.saliency = np.ascontiguousarray(saliency.astype(float))

        arc_functions = ('exp', 'exp-saliency')
        if arc_fun.lower() not in arc_functions:
            raise ValueError('Arc-weight function not found, must include {}'.format(arc_functions))

        self.arc_fun = arc_fun.lower()

        if self.arc_fun.startswith('exp'):
            sigma = 1.0
            if 'sigma' not in kwargs:
                warnings.warn('`sigma` not provided, using default, %f' % sigma, Warning)
            self.sigma = kwargs.pop('sigma', sigma)

        if self.arc_fun == 'exp-saliency':
            if saliency is None:
                raise TypeError('`saliency` must be provided with `exp-saliency` arc-weight.')
            gamma = 1.0
            if 'gamma' not in kwargs:
                warnings.warn('`gamma` not provided, using default %f' % gamma, Warning)
            self.gamma = kwargs.pop('gamma', gamma)

        self.size = image.shape[:2]
        self.image = np.ascontiguousarray(image.astype(float))
        self.costs = np.full(self.size, np.finfo('d').max, dtype=float)
        self.preds = np.full(self.size, -1, dtype=int)
        self.labels = np.zeros(self.size, dtype=bool)

        self.source = -1
        self.destiny = -1
        self.start = -1
        self.paths = collections.OrderedDict()
        self.current = None

    def _opt_path(self, src: int, dst: int) -> Optional[np.ndarray]:
        """
        Compute optimum-path from source to destiny.

        Parameters
        ----------
        src : int
            Source index.
        dst : int
            Destiny index.

        Returns
        -------
        array_like
            Array of flattened indices.
        """
        if self.arc_fun == 'exp':
            path = _pyift.livewire_path(self.image, self.costs, self.preds, self.labels,
                                        self.arc_fun, self.sigma, src, dst)
        elif self.arc_fun == 'exp-saliency':
            path = _pyift.livewire_path(self.image, self.saliency, self.costs, self.preds, self.labels,
                                        self.arc_fun, self.sigma, self.gamma, src, dst)
        else:
            raise NotImplementedError

        return path

    def _assert_valid(self, y: int, x: int) -> None:
        """
        Asserts coordinates belong in the image domain.
        """
        if not (0 <= y < self.size[0] and 0 <= x < self.size[1]):
            raise ValueError('Coordinates out of image boundary, {}'.format(self.size))

    def select(self, position: Union[Sequence[int], int]) -> None:
        """
        Selects next position to compute optimum-path to, or initial position.

        Parameters
        ----------
        position: Sequence[int, int], int
            Index or coordinate (y, x) in the image domain
        """
        if isinstance(position, (list, tuple, np.ndarray)):
            y, x = round(position[0]), round(position[1])
            self._assert_valid(y, x)
            position = int(y * self.size[1] + x)

        if not isinstance(position, int):
            raise TypeError('`position` must be a integer, tuple or list.')

        if self.source != -1:
            self.cancel()
            self.current = self._opt_path(self.source, position)
        else:
            self.start = position

        self.destiny = position  # must be after cancel

    def cancel(self) -> None:
        """
        Cancel current unconfirmed path.
        """
        if self.current is not None and self.current.size:
            # reset path
            self.labels.flat[self.current] = False
            self.costs.flat[self.current] = np.finfo('d').max
            # reset path end
            self.labels.flat[self.destiny] = False
            self.costs.flat[self.destiny] = np.finfo('d').max

    def confirm(self) -> None:
        """
        Confirms current path and sets it as new path source.
        """
        if self.source != -1:
            self.paths[self.source] = self.current
        self.source = self.destiny
        self.current = None

    def close(self) -> None:
        """
        Connects the current path to the initial coordinate, closing the live-wire contour. Result must be confirmed.
        """
        if len(self.paths) == 0:
            raise ValueError('Path must be confirmed before closing contour')
        self.cancel()
        self.costs.flat[self.start] = np.finfo('d').max
        self.select(self.start)
        self.confirm()
        self.start = -1
        self.source = -1
        self.destiny = -1

    @property
    def contour(self) -> np.ndarray:
        """
        Returns
        -------
        array_like
            Optimum-path contour.
        """
        return self.labels
