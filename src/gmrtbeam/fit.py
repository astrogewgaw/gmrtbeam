from typing import Literal
from dataclasses import dataclass

import numpy as np
from lmfit import Model
import ultraplot as uplt
from cv2 import fitEllipse
from skimage.measure import find_contours

from gmrtbeam.core import GMRTBeam


def gaussian(
    xy: np.ndarray,
    A: float,
    Wx: float,
    Wy: float,
    theta: float,
    x0: float = 0.0,
    y0: float = 0.0,
):
    x, y = xy
    theta = np.deg2rad(theta)
    xpolar = x * np.cos(theta) - y * np.sin(theta)
    ypolar = x * np.sin(theta) + y * np.cos(theta)
    x0polar = x0 * np.cos(theta) - y0 * np.sin(theta)
    y0polar = x0 * np.sin(theta) + y0 * np.cos(theta)

    X = (x0polar - xpolar) / Wx
    Y = (y0polar - ypolar) / Wy
    Z = A * np.exp(-1 * (X**2 + Y**2) / 2.0)
    return Z.ravel()


@dataclass
class BeamFitter:
    beam: GMRTBeam
    contour: tuple[np.ndarray, np.ndarray] | None = None
    model: Literal["gaussian", "elliptical"] = "gaussian"

    def fit(
        self,
        box: int,
        fix: bool = False,
        isophot: float = 0.5,
    ):
        data = self.beam.data
        if data is None:
            self.beam.compute()
            data = self.beam.data
            assert data is not None

        npix, npix = data.shape
        fovsize = self.beam.fovsize
        deltafov = fovsize / npix
        beamwidth = self.beam.size / deltafov
        xside = yside = np.ceil(box * beamwidth).astype(int)
        xc, yc = np.unravel_index(np.argmax(data), data.shape)

        match self.model:
            case "gaussian":
                cutout = data[xc - xside : xc + xside, yc - yside : yc + yside]

                A = 1.0
                theta = 0.0
                x0, y0 = xside, yside
                Nx, Ny = cutout.shape
                Wx = Wy = beamwidth / 1.5
                X, Y = np.arange(Nx), np.arange(Ny)
                xy = np.vstack((X.ravel(), Y.ravel()))

                model = Model(gaussian)
                params = model.make_params(
                    A=dict(value=A, min=0.0, max=1.0),
                    Wx=dict(value=Wx, min=0.0, max=np.inf),
                    Wy=dict(value=Wy, min=0.0, max=np.inf),
                    theta=dict(value=theta, min=0.0, max=360.0),
                    x0=dict(value=x0, min=0.0, max=2 * xside, vary=fix),
                    y0=dict(value=y0, min=0.0, max=2 * xside, vary=fix),
                )

                result = model.fit(cutout, params, xy=xy)
                bestvals = result.best_values

                theta = bestvals["theta"]
                x0, y0 = bestvals["x0"], bestvals["y0"]
                Wx, Wy = bestvals["Wx"], bestvals["Wy"]
                x0, y0 = x0 + xc - xside, y0 + yc - yside
                theta = theta - np.floor(theta / 360) * 360
                theta = theta - 180.0 if theta > 180.0 else theta
                if Wx > Wy:
                    a, b = Wx, Wy
                    theta = theta + 90
                    theta = theta - 180.0 if theta > 180.0 else theta
                else:
                    b, a = Wx, Wy
                a = a * np.sqrt(-np.log(isophot) * 2)
                b = b * np.sqrt(-np.log(isophot) * 2)
                t = np.deg2rad(theta + 90)
                e = np.sqrt(1.0 - b * b / (a * a))
                x = np.zeros(360, dtype=np.float32)
                y = np.zeros(360, dtype=np.float32)

                for i, p in enumerate(np.linspace(0, 2 * np.pi, 360)):
                    r = b / np.sqrt(1 - (e * np.cos(p + t)) ** 2)
                    x[i] = r * np.cos(p) + x0 - npix / 2
                    y[i] = r * np.sin(p) + y0 - npix / 2
                x = x * np.rad2deg(deltafov) * 3600
                y = y * np.rad2deg(deltafov) * 3600
                self.contour = x, y

            case "elliptical":
                params = (0, 0), (-1, -1), 0

                found = False
                maxoffset = 0.2 * beamwidth
                contours = find_contours(data, isophot)
                for contour in contours:
                    contour = np.array(contour, dtype=np.float32)
                    ellipse = (0, 0), (-1, -1), 0
                    ellipse = fitEllipse(contour)
                    (x0i, y0i), (_, ai), _ = ellipse
                    xoffset, yoffset = np.abs(x0i - xc), np.abs(y0i - yc)
                    if (xoffset < maxoffset) and (yoffset < maxoffset):
                        if not found:
                            found = True
                            params = ellipse
                        else:
                            (x0, y0), (b, a), t = params
                            params = ellipse if a > ai else params
                if found:
                    (x0, y0), (b, a), t = params

                    t = np.deg2rad(t)
                    e = np.sqrt(1.0 - b * b / (a * a))
                    x = np.zeros(360, dtype=np.float32)
                    y = np.zeros(360, dtype=np.float32)

                    for i, p in enumerate(np.linspace(0, 2 * np.pi, 360)):
                        r = 0.5 * b / np.sqrt(1 - (e * np.cos(p + t)) ** 2)
                        x[i] = r * np.cos(p) + x0 - npix / 2
                        y[i] = r * np.sin(p) + y0 - npix / 2
                    x = x * np.rad2deg(deltafov) * 3600
                    y = y * np.rad2deg(deltafov) * 3600
                    self.contour = x, y

    def plot(self, ax: uplt.Axes | None = None):
        if self.contour is not None:
            if ax is None:
                fig = uplt.figure(width=10, height=10)
                ax = fig.subplot()  # type: ignore
                assert ax is not None
            self.beam.plot(ax=ax, show=False)
            ax.plot(*self.contour)
            uplt.show()
