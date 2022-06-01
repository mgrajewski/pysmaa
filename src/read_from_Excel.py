# -*- coding: utf-8 -*-
"""
@author: M. Grajewski, E. Bertelsmann, FH Aachen University of Applied Sciences

This module provides functions for reading the MCDA data from an Excel file.
"""
import numpy as np
import openpyxl as oxl


def read_ptilde_from_excel(file_name):
    """
    This function reads the raw performance matrix Ptilde from an Excel file.
    It is assumed that the performance matrix is contained in a worksheet
    called "Characteristics". This sheet title is case-sensitive.
    Mean values of Ptilde are indicated by the keyword "Mean" in the Excel
    worksheet, upper bounds by "Max" and lower bounds by "Max". If a keyword is
    provided, then it is required to be in somewhere in the first column of
    that worksheet. In the row below a keyword, starting in the second column,
    the function expects the names of the criteria. In the second column,
    starting two rows below the keyword, the function expects the names of the
    alternatives:

    .. math::
        \\begin{bmatrix}
            \\text{keyword} & \\text{ } & \\text{ } & \\text{ }\\\\
            \\text{criteria} & \\text{1st crit} & \\cdots & \\text{last crit}\\\\
            \\text{1st alt} & p & p & p\\\\
            \\vdots & p & p & p\\\\
            \\text{last alt} & p & p & p
        \\end{bmatrix}


    Where :math:`p` represents the performance matrix.

    There must be no empty columns in the part of the Excel sheet with the performance matrix.

    If both "Min" and "Max" exist, "Mean" is optional, if "Mean" exists, then both "Min" and "Max" are optional, but if
    one of these keywords exists, the other must be provided as well.

    The function returns the performance matrix Ptilde, its upper bounds PtildeMax and its lower bounds PtildeMin. If
    the bounds are not provided, it returns PtildeMax = PtildeMin = Ptilde. If mean values are not provided, it returns
    a matrix filled with zeroes for Ptilde.

    Parameters
    ----------
    file_name : string
        name of the Excel file to read from

    Returns
    -------
    Ptilde : 2D-numpy-array
    PtildeMin : 2D-numpy-array
    PtildeMax : 2D-numpy-array
        raw performance matrix with optional upper and lower bounds

    criteria : list containing the names of the criteria
    alternatives:  list containing the names of the alternatives

    """
    # tolerance for comparing floating point numbers
    tol = 1e-12

    workbook = oxl.load_workbook(filename=file_name, data_only=True)

    # maximal row to consider at all (worksheet.max_row is sometimes unreliable)
    max_row = 200

    # maximal column to consider at all
    max_column = 100

    row_Min = max_row
    row_Max = max_row

    # open Excel worksheet with name "Characteristics'
    worksheet = workbook['Characteristics']

    # maximal column with non-empty cells (as worksheet.max_column is
    # unreliable, we safeguard it with the initial max_column). The same
    # applies to max_row
    max_column = max(max_column, worksheet.max_column)
    max_row = max(max_row, worksheet.max_row)

    # find keyword "Min" in first column
    min_found = False
    for irow in range(1, max_row):
        if worksheet.cell(row=irow, column=1).value == 'Min':
            row_Min = irow
            min_found = True
            break

    # find keyword "Max" in first column
    max_found = False
    for irow in range(1, max_row):
        if worksheet.cell(row=irow, column=1).value == 'Max':
            row_Max = irow
            max_found = True
            break

    # find keyword "Mean" in first column
    mean_found = False
    for irow in range(1, max_row):
        if worksheet.cell(row=irow, column=1).value == 'Mean':
            row_Mean = irow
            mean_found = True
            break

    # all three matrices provided
    if min_found and max_found and mean_found:

        istart = min(row_Mean, row_Min, row_Max)
        if istart == row_Mean:
            istop = min(row_Min, row_Max)
        elif istart == row_Min:
            istop = min(row_Mean, row_Max)
        elif istart == row_Max:
            istop = min(row_Mean, row_Min)

    # only mean value provided
    elif mean_found and not min_found and not max_found:
        istart = row_Mean
        istop = row_Max

    # no matrix provided at all
    elif not mean_found and not min_found and not max_found:
        err_msg = 'Neither Mean values nor Min values nor Max values ' + \
                  'provided for Ptilde.'
        raise NameError(err_msg)

    # only upper and lower bounds provided
    elif min_found and max_found and not mean_found:
        istart = min(row_Min, row_Max)
        istop = max(row_Min, row_Max)

    # all other cases
    else:
        err_msg = 'Please provide either mean values for Ptilde ' + \
                  'or upper and lower bounds or all of this.'
        raise NameError(err_msg)

    # find number of alternatives by considering the first of the three
    # matrices found in the Excel sheet
    nalts = 0
    for irow in range(istart+2, istop):
        if worksheet.cell(row=irow, column=1).value is not None:
            nalts = nalts+1

    # find number of criteria (first guess)
    ncrit = max_column-1

    criteria = []
    alternatives = []

    # first row after the first keyword
    if mean_found:
        row_read = row_Mean
    elif min_found:
        row_read = row_Min
    elif max_found:
        row_read = row_Max

    # read in the first column after the first keyword found (contains the
    # names of the criteria)
    for icol in range(2, ncrit+2):
        if worksheet.cell(row=1+row_read, column=icol).value is not None:
            criteria.append(worksheet.cell(row=1+row_read, column=icol).value)

    # this is the real number of criteria (it could happen that there are some filled Excel cells apart from the actual
    # data such that max_column-1 is greater than the number of criteria)
    ncrit = len(criteria)

    # allocate matrices and initialise them with zeros
    Ptilde = np.zeros((nalts, ncrit))
    PtildeMin = np.zeros((nalts, ncrit))
    PtildeMax = np.zeros((nalts, ncrit))

    # read the names of the alternatives
    for irow in range(row_read+2, nalts+3):
        if worksheet.cell(row=irow, column=1).value is not None:
            alternatives.append(worksheet.cell(row=irow, column=1).value)
        else:
            err_msg = 'Reading the names of the alternatives failed due ' + \
                      'to empty cells. Please name every alternative.'
            raise NameError(err_msg)

    if mean_found:
        # read mean values of the performance matrix
        for irow in range(0, nalts):
            for icol in range(0, ncrit):
                if worksheet.cell(row=irow+3, column=icol+2).value is not None:
                    Ptilde[irow, icol] = worksheet.cell(row=irow+3, column=icol+2).value
                else:
                    err_msg = 'Reading the mean values for Ptilde failed ' + \
                              'due to empty cells (row ' + str(irow+3) + \
                              ', column ' + str(icol+2)+').'
                    raise NameError(err_msg)

    # if lower bounds are provided, read them
    if min_found:
        for irow in range(0, nalts):
            for j in range(0, ncrit):
                if worksheet.cell(row=irow+2+row_Min, column=j+2).value is not None:
                    PtildeMin[irow, j] = worksheet.cell(row=irow+2+row_Min, column=j+2).value
                else:
                    err_msg = 'Reading the minimal values for Ptilde ' + \
                              'failed due to empty cells (row ' + \
                              str(irow+2+row_Min) + ', column ' + str(j+2)+').'
                    raise NameError(err_msg)
    else:
        PtildeMin = Ptilde

    # if upper bounds are provided, read them
    if max_found:
        for i in range(0, nalts):
            for j in range(0, ncrit):
                if worksheet.cell(row=i+2+row_Max, column=j+2).value is not None:
                    PtildeMax[i, j] = worksheet.cell(row=i+2+row_Max, column=j+2).value
                else:
                    err_msg = 'Reading the maximal values for Ptilde ' + \
                              'failed due to empty cells (row ' + \
                              str(i+2+row_Max) + ', column ' + str(j+2)+').'
                    raise NameError(err_msg)
    else:
        PtildeMax = Ptilde

    # test, if some components of PtildeMin are greater than PtildeMax
    # Testing if these matrices have been provided ist not necessary, as in
    # case, it holds PtildeMax = PtildeMin = Ptilde, such that there is no
    # difference between these matrices.
    if ((PtildeMax - PtildeMin) < -tol).any():
        raise NameError('At least one lower bound for Ptilde is greater ' +
                        'than the corresponding upper bound of Ptilde.')

    if max_found:
        if ((PtildeMax - Ptilde) < -tol).any():
            raise NameError('At least one upper bound for Ptilde is smaller ' +
                            'than the corresponding mean value of Ptilde.')

    if min_found:
        if (Ptilde - PtildeMin < -tol).any():
            raise NameError('At least one lower bound for Ptilde is greater ' +
                            'than the corresponding mean value of Ptilde.')

    # close the Excel file
    workbook.close()

    return Ptilde, PtildeMin, PtildeMax, criteria, alternatives


def read_rankings_from_excel(file_name):
    """
    This function reads the desired rankings from an Excel file.
    It is assumed that the data are contained in a worksheet
    called "Ranking". This sheet title is case-sensitive. The first row
    is reserved for the names of the stakeholders associated with the rankings,
    which are given column-wise in Excel. The rankings are expected to start
    in the first column.

    Parameters
    ----------
    file_name : string
        name of the Excel file to read from

    Returns
    -------
    rankings : list of 1D-numpy integer arrays
        1-based rankings
    stakeholders : list of strings
        names of the stakeholders associated to the rankings
    """
    workbook = oxl.load_workbook(filename=file_name)

    # open the worksheet "Ranking"
    worksheet = workbook['Ranking']

    # maximal row and column index occurring in this shet
    max_row = worksheet.max_row
    max_column = worksheet.max_column

    stakeholders = []
    rankings = []

    # loop over the first row to get the number of rankings
    nranks = 0
    for icol in range(1, max_column+1):
        if worksheet.cell(row=1, column=icol).value is not None:
            stakeholders.append(worksheet.cell(row=1, column=icol).value)
            nranks = nranks+1
        else:
            break

    for irank in range(0, nranks):
        rank_length = 0
        for irow in range(0, max_row):
            if worksheet.cell(row=irow+2, column=irank+1).value is not None:
                rank_length = rank_length+1
            else:
                break

        rankings.append(np.zeros(rank_length, dtype=int))

        for irow in range(0, rank_length):
            rankings[irank][irow] = worksheet.cell(row=irow+2, column=irank+1).value

    return rankings, stakeholders


def read_bounds_from_excel(file_name):
    """
    This function reads upper and lower bounds for weights from an Excel file for all stakeholders considered.
    It is assumed that the data are contained in a worksheet called bounds. This sheet title is case-sensitive. The
    first row is reserved for the names of the stakeholders associated with the rankings, which are given column-wise
    in Excel.
    It is assumed in the Excel file, that any of the weights are bounded and that the order of weights corresponds to
    their order of the performance matrix. As the sum of weights is one, providing an upper bound of 1 and a lower
    bound of 0 corresponds to no constraints.
    For additional flexibility, upper and lower bounds are reformulated as linear conditions of the shape
    :math:`Aw \\leq b`, :math:`w` being the weight, as this format allows for arbitrary linear constraints
    like e.g. :math:`w_1 < w_2` in the future.

    Parameters
    ----------
    file_name : string
        name of the Excel file to read the bounds from

    Returns
    -------
    All_A_constraints: list of 2D numpy arrays
        All_A_constraints[i] contains the constraints matrix for the i-th stakeholder
    All_b_constraints: list of 1D numpy arrays
        All_b_constraints[i] contains the constraints right-hand side for the i-th stakeholder
    stakeholders : list of strings
        names of the stakeholders associated to the rankings
    """
    eps = 1e-10

    workbook = oxl.load_workbook(filename=file_name, data_only=True)

    # read in the characteristics
    worksheet = workbook['bounds']

    # find keyword "MINVALUES" in first column
    min_found = False
    for irow in range(1, 100):
        if worksheet.cell(row=irow, column=1).value == 'MINVALUE':
            row_Min = irow
            min_found = True

    # find keyword "MAXVALUES" in first column
    max_found = False
    for irow in range(1, 100):
        if worksheet.cell(row=irow, column=1).value == 'MAXVALUE':
            row_Max = irow
            max_found = True

    if not min_found and not max_found:
        raise NameError('Neither upper nor lower bounds provided')

    offset = row_Min+2

    # get number of criteria by counting the non-empty cells in the first column
    ncrit = 0
    for i in range(offset, worksheet.max_row):
        if worksheet.cell(column=1, row=i).value is not None:
            ncrit = ncrit + 1
        else:
            break

    All_A_constraints = []
    All_b_constraints = []
    stakeholders = []

    # read in the column below the keyword "MINVALUE" (contains the names of
    # the stakeholders) and get the number of stakeholders
    for icolumn in range(2, worksheet.max_row):
        if worksheet.cell(row=2, column=icolumn).value is not None:
            stakeholders.append(worksheet.cell(row=2, column=icolumn).value)

    n_stakeholders = len(stakeholders)

    # read in lower bounds

    # run over columns
    for icolumn in range(2, n_stakeholders+2):
        n_constrs = 0

        # consistency check
        for irow in range(offset, ncrit+offset):
            irow_max = irow - offset+row_Max+2
            if abs(float(worksheet.cell(row=irow, column=icolumn).value) -
                   float(worksheet.cell(row=irow_max, column=icolumn).value)) < eps:
                err_msg = 'For stakeholder ' + stakeholders[icolumn-2] + \
                         ', a lower bound equals an upper bound. ' + \
                         'This is not supported yet.'
                raise NameError(err_msg)

            if float(worksheet.cell(row=irow_max, column=icolumn).value) - \
               float(worksheet.cell(row=irow, column=icolumn).value) < -eps:
                err_msg = 'For stakeholder ' + stakeholders[icolumn-2] + \
                         ', a lower bound is larger than an upper bound. '
                raise NameError(err_msg)

        for irow in range(offset, ncrit+offset):
            if float(worksheet.cell(row=irow, column=icolumn).value) > 0:
                n_constrs = n_constrs + 1

        A_constraints = np.zeros((n_constrs, ncrit))
        b_constraints = np.zeros(n_constrs)
        iidx = 0
        for irow in range(offset, ncrit+offset):
            if float(worksheet.cell(row=irow, column=icolumn).value) > 0:
                A_constraints[iidx, irow-offset] = -1.0
                b_constraints[iidx] = \
                    -float(worksheet.cell(row=irow, column=icolumn).value)
                iidx = iidx+1

        All_A_constraints.append(A_constraints)
        All_b_constraints.append(b_constraints)

    offset = row_Max+2

    # read in upper bounds
    for icolumn in range(2, n_stakeholders+2):
        n_constrs = 0
        for irow in range(offset, ncrit+offset):
            if float(worksheet.cell(row=irow, column=icolumn).value) < 1:
                n_constrs = n_constrs + 1

        A_constraints_max = np.zeros((n_constrs, ncrit))
        b_constraints_max = np.zeros(n_constrs)
        iidx = 0
        for irow in range(offset, ncrit+offset):
            if float(worksheet.cell(row=irow, column=icolumn).value) < 1.0:
                A_constraints_max[iidx, irow-offset] = 1.0
                b_constraints_max[iidx] = \
                    float(worksheet.cell(row=irow, column=icolumn).value)
                iidx = iidx+1

        All_A_constraints[icolumn-2] = \
            np.vstack((All_A_constraints[icolumn-2], A_constraints_max))

        All_b_constraints[icolumn-2] = \
            np.hstack((All_b_constraints[icolumn-2], b_constraints_max))

    return All_A_constraints, All_b_constraints, stakeholders


def read_weights_from_excel(file_name):
    """
    This function reads the weights of criteria from an Excel file. It is assumed that the data are contained in a
    worksheet called "Weights". This sheet title is case-sensitive. The first row is reserved for the names of the
    stakeholders associated with the weights, which are given column-wise in Excel. The weights are expected to start
    in the first column.

    Parameters
    ----------
    file_name : string
        name of the Excel file to read from

    Returns
    -------
    weights : list of 1D-numpy integer arrays
        1-based rankings
    stakeholders : list of strings
        names of the stakeholders associated to the weights
    """
    workbook = oxl.load_workbook(filename=file_name)

    # open the worksheet "Ranking"
    worksheet = workbook['Weights']

    # maximal row and column index occurring in this shet
    max_row = worksheet.max_row
    max_column = worksheet.max_column

    criteria = []
    stakeholders = []
    weights = []

    # loop over the first row to get the number of stakeholders and their names
    nstakeholders = 0
    for icol in range(2, max_column+1):
        if worksheet.cell(row=1, column=icol).value is not None:
            stakeholders.append(worksheet.cell(row=1, column=icol).value)
            nstakeholders = nstakeholders+1
        else:
            break

    # loop over the first column to get the number of criteria and their names
    ncrit = 0
    for irow in range(0, max_row):
        if worksheet.cell(row=irow+2, column=1).value is not None:
            criteria.append(worksheet.cell(row=irow+2, column=1).value)
            ncrit = ncrit + 1
        else:
            break

    for iweight in range(0, nstakeholders):
        weights.append(np.zeros(ncrit, dtype=float))

        for irow in range(0, ncrit):
            weights[iweight][irow] = worksheet.cell(row=irow+2, column=iweight+2).value

    return criteria, stakeholders, weights
