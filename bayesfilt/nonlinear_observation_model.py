"""Classes for defining linear observation models """
import numpy as np
from numpy import ndarray
from .observation_model import ObservationModel


class CTRAobs1(ObservationModel):
    # pylint: disable=invalid-name
    """Class for defining a linear observation model"""

    def __init__(
        self,
    ) -> None:

        super().__init__(nx=6, ny=2, name='CTRA-TEL2')
        self._H: ndarray = np.zeros((self._ny, self._nx))  # obs function

    def update(
        self,
        sigmas: ndarray | None = None,
        R: ndarray | None = None,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        if R is not None:
            self._R = self.mat_setter(R, (self.ny, self.ny))
        else:
            self._sigmas = self.vec_setter(sigmas, self.ny)
            self._R = np.diag(self.sigmas**2)
        self._J = np.eye(self.ny)

    def h(
        self,
        x: ndarray | None,
        r: ndarray | None = None
    ) -> ndarray:
        """Measurement equation"""
        x = self.vec_setter(x, self.nx)
        out_y = np.zeros((self.ny,))
        out_y[0] = x[0]
        out_y[1] = x[1]
        # out_y[2] = np.arctan2(x[2], x[3]) % (2.0 * np.pi)
        # out_y[3] = np.sqrt(x[2]**2 + x[3]**2)
        if r is not None:
            r = self.vec_setter(r, self.ny)
            out_y += r
        return out_y

    def compute_H(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get H matrix"""
        return self.H

    def compute_J(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get J matrix"""
        return self.J


class CTRAobs2(ObservationModel):
    # pylint: disable=invalid-name
    """Class for defining a linear observation model"""

    def __init__(
        self,
    ) -> None:

        super().__init__(nx=6, ny=4, name='CTRA-TEL3')
        self._H: ndarray = np.zeros((self._ny, self._nx))  # obs function

    def update(
        self,
        sigmas: ndarray | None = None,
        R: ndarray | None = None,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        if R is not None:
            self._R = self.mat_setter(R, (self.ny, self.ny))
        else:
            self._sigmas = self.vec_setter(sigmas, self.ny)
            self._R = np.diag(self.sigmas**2)
        self._J = np.eye(self.ny)

    def h(
        self,
        x: ndarray | None,
        r: ndarray | None = None
    ) -> ndarray:
        """Measurement equation"""
        x = self.vec_setter(x, self.nx)
        out_y = np.zeros((self.ny,))
        out_y[0] = x[0]
        out_y[1] = x[1]
        out_y[2] = np.arctan2(x[2], x[3]) % (2.0 * np.pi)
        out_y[3] = np.sqrt(x[2]**2 + x[3]**2)
        if r is not None:
            r = self.vec_setter(r, self.ny)
            out_y += r
        return out_y

    def compute_H(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get H matrix"""
        return self.H

    def compute_J(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get J matrix"""
        return self.J


class CA2D_TEL1(ObservationModel):
    # pylint: disable=invalid-name
    """Class for defining a linear observation model"""

    def __init__(
        self,
    ) -> None:

        super().__init__(nx=6, ny=4, name='CA2DTEL3')
        self._H: ndarray = np.zeros((self._ny, self._nx))  # obs function

    def update(
        self,
        sigmas: ndarray | None = None,
        R: ndarray | None = None,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        if R is not None:
            self._R = self.mat_setter(R, (self.ny, self.ny))
        else:
            self._sigmas = self.vec_setter(sigmas, self.ny)
            self._R = np.diag(self.sigmas**2)
        self._J = np.eye(self.ny)

    def h(
        self,
        x: ndarray | None,
        r: ndarray | None = None
    ) -> ndarray:
        """Measurement equation"""
        x = self.vec_setter(x, self.nx)
        out_y = np.zeros((self.ny,))
        out_y[0] = x[0]
        out_y[1] = x[3]
        out_y[2] = np.arctan2(x[1], x[4]) % (2.0 * np.pi)
        out_y[3] = np.sqrt(x[1]**2 + x[4]**2)
        if r is not None:
            r = self.vec_setter(r, self.ny)
            out_y += r
        return out_y

    def compute_H(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get H matrix"""
        return self.H

    def compute_J(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get J matrix"""
        return self.J


class CA2D_TEL2(ObservationModel):
    # pylint: disable=invalid-name
    """Class for defining a linear observation model"""

    def __init__(
        self,
    ) -> None:

        super().__init__(nx=6, ny=2, name='CA2DTEL3')
        self._H: ndarray = np.zeros((self._ny, self._nx))  # obs function

    def update(
        self,
        sigmas: ndarray | None = None,
        R: ndarray | None = None,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        if R is not None:
            self._R = self.mat_setter(R, (self.ny, self.ny))
        else:
            self._sigmas = self.vec_setter(sigmas, self.ny)
            self._R = np.diag(self.sigmas**2)
        self._J = np.eye(self.ny)

    def h(
        self,
        x: ndarray | None,
        r: ndarray | None = None
    ) -> ndarray:
        """Measurement equation"""
        x = self.vec_setter(x, self.nx)
        out_y = np.zeros((self.ny,))
        out_y[0] = x[0]
        out_y[1] = x[3]
        # out_y[2] = np.arctan2(x[1], x[4]) % (2.0 * np.pi)
        # out_y[3] = np.sqrt(x[1]**2 + x[4]**2)
        if r is not None:
            r = self.vec_setter(r, self.ny)
            out_y += r
        return out_y

    def compute_H(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get H matrix"""
        return self.H

    def compute_J(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get J matrix"""
        return self.J


class CA3D_TEL(ObservationModel):
    # pylint: disable=invalid-name
    """Class for defining a linear observation model"""

    def __init__(
        self,
    ) -> None:

        super().__init__(nx=9, ny=5, name='CA2DTEL3')
        self._H: ndarray = np.zeros((self._ny, self._nx))  # obs function

    def update(
        self,
        sigmas: ndarray | None = None,
        R: ndarray | None = None,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        if R is not None:
            self._R = self.mat_setter(R, (self.ny, self.ny))
        else:
            self._sigmas = self.vec_setter(sigmas, self.ny)
            self._R = np.diag(self.sigmas**2)
        self._J = np.eye(self.ny)

    def h(
        self,
        x: ndarray | None,
        r: ndarray | None = None
    ) -> ndarray:
        """Measurement equation"""
        x = self.vec_setter(x, self.nx)
        out_y = np.zeros((self.ny,))
        out_y[0] = x[0]
        out_y[1] = x[3]
        out_y[2] = x[6]
        out_y[3] = np.arctan2(x[1], x[4]) % (2.0 * np.pi)
        out_y[4] = np.sqrt(x[1]**2 + x[4]**2)
        if r is not None:
            r = self.vec_setter(r, self.ny)
            out_y += r
        return out_y

    def compute_H(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get H matrix"""
        return self.H

    def compute_J(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get J matrix"""
        return self.J
