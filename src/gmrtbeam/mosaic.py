from pathlib import Path
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import ultraplot as uplt
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from ultraplot.axes import Axes
from astropy.visualization.wcsaxes import SphericalCircle

from gmrtbeam.core import GMRTBeam
from gmrtbeam.fit import BeamFitter, BeamEllipse


@dataclass
class GMRTMosaic:
    nbeams: int
    beam: GMRTBeam
    fitter: BeamFitter
    radius: float = 0.0
    ellipse: BeamEllipse | None = None
    center: tuple[float, float] = (0.0, 0.0)
    grid: tuple[np.ndarray, np.ndarray] | None = None

    def to_fits(self, fn: str | Path):
        if self.grid is not None:
            header = self.wcs.to_header()
            hdu = fits.PrimaryHDU(header=header)
            hdu.data = self.grid
            hdu.update_header()
            hdu.writeto(fn, overwrite=True)

    @cached_property
    def npix(self) -> int:
        return len(self.grid[0]) if self.grid is not None else self.nbeams

    @cached_property
    def wcs(self) -> WCS:
        dra = ddec = 2.2 * np.rad2deg(self.radius) / self.npix
        dra = -dra
        return WCS(
            {
                "CTYPE1": "RA---SIN",
                "CTYPE2": "DEC--SIN",
                "CUNIT1": "deg",
                "CUNIT2": "deg",
                "CDELT1": dra,
                "CDELT2": ddec,
                "NAXIS1": self.npix,
                "NAXIS2": self.npix,
                "CRPIX1": self.npix / 2,
                "CRPIX2": self.npix / 2,
                "CRVAL1": self.beam.ra.deg,
                "CRVAL2": self.beam.dec.deg,
                "PC1_1 ": 1.0,
                "PC1_2 ": 0.0,
                "PC2_1 ": 0.0,
                "PC2_2 ": 1.0,
            }
        )

    def tile(
        self,
        eff: float = 0.9,
        fac: float = 1.05,
        dgrid0: float = 0.05,
        maxiterin: int = 50,
        maxiterout: int = 10,
        beamusage: float = 0.95,
        areausage: float = 0.85,
        recenter: bool = True,
    ):
        self.fitter.fit()
        self.ellipse = self.fitter.contour
        if self.ellipse is not None:

            def gridder(N: int, R: float, ellipse: BeamEllipse, recenter: bool = True):
                t = ellipse.t
                t = np.arctan2(np.sin(t), -np.cos(t))
                N2 = 4 * np.floor(R / ellipse.b).astype(int)
                N1 = 4 * np.floor(R / (np.sqrt(3) / 2 * ellipse.a)).astype(int)
                X, Y = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)

                dx, dy = np.sqrt(3) / 2 * ellipse.a, ellipse.b
                x = -R - 0.5 * dx
                i = 0
                for ii in range(N1):
                    y = -R - 0.5 * dy
                    y = y + 0.5 * dy if ii % 2 else y
                    for _ in range(N2):
                        r = np.sqrt(x * x + y * y)
                        if r < R:
                            X[i], Y[i] = x, y
                            i = i + 1
                            if i == N:
                                break
                        y = y + dy
                    x = x + dx
                    if i == N:
                        break
                XPOLAR = X * np.cos(t) - Y * np.sin(t)
                YPOLAR = X * np.sin(t) + Y * np.cos(t)
                if recenter:
                    radii = np.sqrt(XPOLAR[:i] * XPOLAR[:i] + YPOLAR[:i] * YPOLAR[:i])
                    icenter = np.argmin(radii)
                    X0, Y0 = XPOLAR[icenter], YPOLAR[icenter]
                    XPOLAR = XPOLAR - X0
                    YPOLAR = YPOLAR - Y0
                    return (XPOLAR, YPOLAR), i, (-X0, -Y0)
                return (XPOLAR, YPOLAR), i, (0, 0)

            N = self.nbeams
            X0, Y0 = 0.0, 0.0
            x, y = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)
            R = 0.5 * float(np.sqrt((N * self.ellipse.a * self.ellipse.b) / eff))

            i = 0
            Nopt = 0
            dgrid = dgrid0
            increase = False
            outconverge = False
            while not outconverge:
                j = 0
                nt0 = 0
                Ropt = R
                ntmax = 0
                inconverge = False
                n = np.floor(N * fac).astype(int)
                while not inconverge:
                    (x, y), Nopt, (X0, Y0) = gridder(
                        R=R,
                        N=self.nbeams,
                        recenter=recenter,
                        ellipse=self.ellipse,
                    )
                    farea = Nopt * (self.ellipse.a * self.ellipse.b) / (4 * R * R)
                    if Nopt == n:
                        if np.sqrt(farea) > 0.95:
                            R = 0.95 * R
                        else:
                            dgrid = dgrid0
                            R = float(np.sqrt(farea)) * R
                    else:
                        if farea > areausage:
                            if Nopt > ntmax:
                                Ropt = R
                                ntmax = Nopt
                        if Nopt >= nt0:
                            if (farea > areausage) and (Nopt >= beamusage * n):
                                inconverge = True
                                break
                        if increase and (Nopt > 0.7 * n):
                            dgrid = 0.9 * dgrid
                        else:
                            increase = True
                            dgrid = dgrid0
                            R = (1 + dgrid) * R
                    nt0 = Nopt
                    j = j + 1
                    if j > maxiterin:
                        break
                if not inconverge:
                    R = Ropt
                    (x, y), Nopt, (X0, Y0) = gridder(
                        R=R,
                        N=self.nbeams,
                        recenter=recenter,
                        ellipse=self.ellipse,
                    )
                if Nopt < N:
                    fac = fac + 0.05
                else:
                    outconverge = True
                    break
                i = i + 1
                if i > maxiterout:
                    break
            if Nopt > N:
                D = np.sqrt((x[:Nopt] - X0) ** 2 + (y[:Nopt] - Y0) ** 2)
                distant = (np.argsort(D)[N:Nopt]).tolist()
                X, Y = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)

                i = 0
                for ix in range(Nopt):
                    if ix not in distant:
                        X[i], Y[i] = x[i], y[i]
                        i = i + 1
                Nopt = N
            else:
                X, Y = x, y
            farea = Nopt * (self.ellipse.a * self.ellipse.b) / (4 * R * R)

            self.radius = R
            self.grid = (X, Y)
            self.center = (X0, Y0)

    def plot(
        self,
        ax: Axes | None = None,
        show: bool = True,
        save: str | None = None,
        **kwargs,
    ):
        def plotter(ax):
            if (self.grid is not None) and (self.ellipse is not None):
                transform = ax.get_transform("icrs")  # type: ignore
                for x, y in zip(*self.grid):
                    ctrcoords = self.beam.coords.spherical_offsets_by(
                        x * u.arcsecond,
                        y * u.arcsecond,
                    )
                    ctrra = getattr(ctrcoords, "ra")
                    ctrdec = getattr(ctrcoords, "dec")
                    ax.scatter(ctrra, ctrdec, color="red", transform=transform)

                    ellcoords = self.beam.coords.spherical_offsets_by(
                        (self.ellipse.x + x) * u.arcsecond,
                        (self.ellipse.y + y) * u.arcsecond,
                    )
                    ellra = getattr(ellcoords, "ra")
                    elldec = getattr(ellcoords, "dec")
                    ax.plot(
                        ellra,
                        elldec,
                        color="black",
                        transform=transform,
                        label="Tiled beams",
                    )

                R = self.radius * u.arcsecond
                circle = SphericalCircle(
                    self.beam.coords,
                    R.to(u.deg),
                    lw=2,
                    edgecolor="r",
                    facecolor="none",
                    transform=transform,
                    label=rf"Tiling radius, $R =$ {R.to(u.arcminute):.2f}",
                )

                ax.plot(
                    self.beam.ra,
                    self.beam.dec,
                    ".",
                    mfc="blue",
                    mec="black",
                    markersize=15,
                    transform=transform,
                    label="Phase center",
                )

                ax.add_patch(circle)

                ax.invert_xaxis()
                ax.set_title("Tiling")
                ax.set_ylabel("Declination (Dec)")
                ax.set_xlabel("Right ascension (RA)")

                lines, labels = [
                    sum(lol, [])
                    for lol in zip(*[ax.get_legend_handles_labels() for ax in fig.axes])
                ]
                unique_labels = set(labels)
                legend = dict(zip(labels, lines))
                unique_lines = [legend[x] for x in unique_labels]
                ax.legend(unique_lines, unique_labels, loc="ur", ncol=1)

        if ax is None:
            if self.grid is not None:
                fig = getattr(uplt, "figure")(figsize=(7.5, 7.5))
                plotter(fig.subplot(projection=self.wcs))
                if show:
                    getattr(uplt, "show")()
                if save is not None:
                    fig.savefig(save, dpi=kwargs.get("dpi", 150))
        else:
            plotter(ax)
