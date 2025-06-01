"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import sys
from os import path
import time

if __name__ == '__main__':

    path_to_pysmaa = "..\\"

    # if a path to pysmaa is provided, read it and append if, if not already present in sys.path
    if path_to_pysmaa != '':
        if not any(path.normcase(sp) == path_to_pysmaa for sp in sys.path):
            sys.path.append(f'{path_to_pysmaa}/src')

import numpy as np
import numpy.typing as npt
import datetime


def export2vtu(points: npt.ArrayLike, triang: npt.ArrayLike, scalar_data: npt.ArrayLike, scalar_data_names: list,
               vec_data: npt.ArrayLike = None, vec_data_names: list = None, filename: str = None) -> None:
    """
    export2vtu exports point-wise data to a vtu-file which can be read in by e.g. Paraview. The function works in 2D
    or 3D. In 2D, triangular and quadrilateral meshes are supported, in 3D tetrahedral and hexahedral meshes for volumes
    and triangular meshes for surfaces. The mesh may be unstructured. Mixed meshes are supported.

    Args:
        points (npt.ArrayLike):
            The set of points of the triangulation. n_verts stands for the number of data points and ndim \in {2, 3} for
            the dimension of the data. Shape: (n_verts, ndim)
        triang (npt.ArrayLike):
            represents the triangulation corresponding to points. The indices of the points are 0-based; the last
            column in this array contains the number of points of the current cell. Shape: (n_cells, MaxPointsPerCell+1)
        scalar_data (npt.ArrayLike):
            n_verts stands for the number of data; points and n_scalar_data_sets represents the number of data sets in
            the vtu-file. Shape: (n_verts x n_scalar_data_sets)
        scalar_data_names:
            list containing the names of the data sets which appear in the vtu-file
        filename (str):
            the filename of the vtu-file. Appends the ending .vtu if it's missing.
            Defaults to export2vtu_ with datetime appended.
        vec_data (n_scalar_data_sets x n_verts x 3, optional): data sets for vector-valued quantities.
            Paraview requires 3D vectors even if the vector field is 2D.
            Note: This order of the shape happens due tue the numpy representation. The first shape parameter
            represents the depth
        vec_data_names (optional): name of the optional vector-valued quantities

    Returns:
        None
    """
    np.set_printoptions(threshold=20, edgeitems=10, linewidth=140,
                        formatter=dict(float=lambda x: "%.3g" % x))  # float arrays %.3g
    if filename is None:
        filename = f"export2VTU_{datetime.now().strftime('%d%m%y_%H%M%S')}.vtu"

    # default: no vector data sets
    n_vec_data_sets = 0

    if vec_data is not None and vec_data_names is not None:
        # determine the number of vector data sets to write
        dim_vec_data_set = len(vec_data.shape)
        n_vec_data_sets = 1 if dim_vec_data_set == 2 else vec_data.shape[0]

        if vec_data_names.shape[0] != n_vec_data_sets:
            print(
                f"The number of vec data sets ({n_vec_data_sets}) does not match the number of names for data "
                f"sets.")
            exit(-1)

    if len(triang.shape) == 1:
        triang = triang.reshape(1, -1)
        print("triang is a 1D array. It gets converted into a 2D (n x 1) array to continue")

    if len(scalar_data.shape) == 1:
        scalar_data = scalar_data.reshape(scalar_data.shape[0], -1)
        print("PointData is a 1D array. It gets converted into a 2D (n x 1) array to continue")

    # determine the dimension of data  space
    ndim = points.shape[1]

    # determine the number of point data sets to write
    n_scalar_data_sets = scalar_data.shape[1]

    if len(scalar_data_names) != n_scalar_data_sets:
        print(
            f"The number of point data sets ({n_scalar_data_sets}) does not match the number of names for data sets.")
        exit(-1)

    # number of vertices
    n_verts = len(scalar_data[:, 0])

    # number of cells in the triangulation
    n_cells, aux = triang.shape

    # points per cell in triangulation
    points_per_cell = triang[:, aux - 1]
    max_points_per_cell = aux

    zcomponent = np.zeros((n_verts, 1)) if ndim == 2 else points[:, 2]

    to_write = ""
    # header of the vtu-file
    to_write += f'<?xml version="1.0"?>\n' \
                f'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" ' \
                f'compressor="vtkZLibDataCompressor">\n' \
                f'  <UnstructuredGrid>\n' \
                f'    <Piece NumberOfPoints="{n_verts}" NumberOfCells="{n_cells}">\n'

    # data points
    to_write += f'      <Points>\n' \
                f'        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'

    for p0, p1, zc in zip(points[:, 0], points[:, 1], zcomponent.reshape(1, -1).ravel()):
        to_write += '          {:.6e} {:.6e} {:.6e}\n'.format(p0, p1, zc)

    to_write += f'        </DataArray>\n' \
                f'      </Points>\n'

    # cell definitions
    to_write += f'      <Cells>\n' \
                f'        <DataArray type="Int32" Name="connectivity" format="ascii">\n'

    polygons = [[] for _ in range(max_points_per_cell)]
    offsets = np.zeros((n_cells, 1))
    poly_types = np.zeros((n_cells, 1))
    iidx_start = 1
    abs_offset = 0

    format_spec_start = '           {:d} {:d} {:d}'
    for i_points_per_cell in range(3, max_points_per_cell):
        # the following two lines are pretty dirty, I know
        polygons[i_points_per_cell] = triang[points_per_cell == i_points_per_cell, :i_points_per_cell]
        n_cells_current_type = polygons[i_points_per_cell].shape[0]
        iidx_end = iidx_start + n_cells_current_type - 1
        if list(range(iidx_start, iidx_end)):
            offsets[iidx_start - 1:iidx_end] = abs_offset + np.array(
                [[i] for i in range(1, n_cells_current_type + 1)]) * i_points_per_cell
            abs_offset = offsets[iidx_end - 1]
        elif iidx_start == iidx_end:
            offsets[iidx_start - 1] = abs_offset + np.array(
                [i + 1 for i in range(n_cells_current_type)]) * i_points_per_cell
            abs_offset = offsets[iidx_start - 1]

        format_spec = format_spec_start + "\n"
        if ndim == 2:
            # VTK_Triangle
            if i_points_per_cell == 3:
                gtype = 5
            # VTK_Line
            elif i_points_per_cell == 2:
                gtype = 3
            # VTK_Quad
            elif i_points_per_cell == 4:
                gtype = 9
            # VTK_Polygon
            else:
                gtype = 7
        else:
            # VTK_Triangle (for surface meshes)
            if i_points_per_cell == 3:
                gtype = 5
            # VTK_Line
            elif i_points_per_cell == 2:
                gtype = 3
            # VTK_Tetra
            elif i_points_per_cell == 4:
                gtype = 10
            # VTK_Hexahedron
            elif i_points_per_cell == 8:
                gtype = 12
            else:
                gtype = 7

        poly_types[iidx_start - 1:iidx_end] = gtype * np.ones((n_cells_current_type, 1))
        if iidx_start == iidx_end:
            poly_types[iidx_start - 1] = gtype * np.ones((n_cells_current_type, 1))
        iidx_start = iidx_end + 1

        if np.any(polygons[i_points_per_cell]):
            for row in polygons[i_points_per_cell]:
                to_write += format_spec.format(*row.astype(int).ravel())
        format_spec_start += " {:d}"
    to_write += f'        </DataArray>\n'

    # offsets
    to_write += f'        <DataArray type="Int32" Name="offsets" format="ascii">\n'
    for offset in offsets:
        to_write += '           {:d}\n'.format(*offset.astype(int).ravel())
    to_write += f'        </DataArray>\n'

    # cell types
    to_write += f'        <DataArray type="UInt8" Name="types" format="ascii">\n'
    for polyType in poly_types:
        to_write += '           {:d}\n'.format(*polyType.astype(int).ravel())
    to_write += f'        </DataArray>\n' \
                f'      </Cells>\n'

    # point data
    to_write += f'      <PointData>\n'
    for i in range(n_scalar_data_sets):
        to_write += f'        <DataArray type="Float32" Name="{scalar_data_names[i]}" NumberOfComponents="1" ' \
                    f'format="ascii">\n'
        for pd in scalar_data[:, i].ravel():
            to_write += '          {:.6e}\n'.format(pd)
        to_write += f"        </DataArray>\n"
    for i in range(n_vec_data_sets):
        to_write += f'        <DataArray type="Float32" Name="{vec_data_names[i]}" NumberOfComponents="3" ' \
                    f'format="ascii">\n'
        for vd in vec_data[i]:
            to_write += '          {:.4e} {:.4e} {:.4e}\n'.format(*vd)
        to_write += '        </DataArray>\n'
    to_write += f'      </PointData>\n'

    # footer section
    to_write += f'    </Piece>\n' \
                f'  </UnstructuredGrid>\n' \
                f'</VTKFile>\n'

    # write the created vtu string to a file
    filename = filename if filename.endswith(".vtu") else filename + ".vtu"
    with open(filename, "w") as f:
        f.write(to_write)


if __name__ == '__main__':
    print('paper_tests_utils')
