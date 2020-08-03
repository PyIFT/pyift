import numpy as np
from pyift.livewire import LiveWire


class TestLiveWire:
    def test_simple_path(self):
        image = np.array([[8, 1, 0, 2, 0],
                          [5, 7, 2, 0, 1],
                          [6, 7, 6, 1, 0],
                          [6, 8, 7, 0, 3],
                          [6, 7, 8, 8, 9]])

        lw = LiveWire(image, sigma=1.0)

        lw.select((0, 0))
        lw.confirm()

        lw.select((4, 4))
        lw.confirm()

        lw.close()

        lw.contour

    def test_saliency_path(self):
        pass

    def test_raises(self):
        image = np.array([[8, 1, 0, 2, 0],
                          [5, 7, 2, 0, 1],
                          [6, 7, 6, 1, 0],
                          [6, 8, 7, 0, 3],
                          [6, 7, 8, 8, 9]])

        with np.testing.assert_raises(TypeError):
            LiveWire(image, saliency=5)

        with np.testing.assert_raises(ValueError):
            LiveWire(image, saliency=image.reshape(-1))

        with np.testing.assert_raises(ValueError):
            LiveWire(image, arc_fun='mistake')

        lw = LiveWire(image, sigma=2.0)

        with np.testing.assert_raises(ValueError):
            lw.close()

        with np.testing.assert_raises(ValueError):
            lw.select((5, 5))

        with np.testing.assert_raises(TypeError):
            lw.select(10.1)
