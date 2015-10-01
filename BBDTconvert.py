from __future__ import division, absolute_import

__author__ = 'Tatiana Likhomanenko'

import sys
import struct
from scipy.special import expit
import numpy

from rep_ef.estimators._matrixnetapplier import MatrixnetClassifier


def unpack_formula(formula_stream, print_=True):
    features = list()  # feature names
    bins_quantities = list()  # bins quantity for each feature
    bins = list()  # list for bins for each feature

    bytes = formula_stream.read(4)
    features_quantity = struct.unpack('i', bytes)[0]
    for index in range(0, features_quantity):
        bytes = formula_stream.read(4)
        factor_length = struct.unpack('i', bytes)[0]
        features.append(formula_stream.read(factor_length))

    bytes = formula_stream.read(4)  # skip formula length
    used_features_quantity = struct.unpack('I', formula_stream.read(4))[0]
    bins_quantities = struct.unpack(
        'I' * used_features_quantity,
        formula_stream.read(4 * used_features_quantity)
    )

    bins_total = struct.unpack('I', formula_stream.read(4))[0]
    if print_:
        print bins_total
    for index in range(used_features_quantity):
        bins.append(
            struct.unpack(
                'f' * bins_quantities[index],
                formula_stream.read(4 * bins_quantities[index])
            )
        )
        if print_:
            print str(features[index]) + " - " + str(bins_quantities[index])
            for j in range(len(bins[index])):
                print bins[index][j]
            print "------------"
    return features, bins_quantities, bins


def convert_lookup_index_to_bins(points_in_bins, lookup_indices):
    result = numpy.zeros([len(lookup_indices), len(points_in_bins)], dtype=float)
    lookup_indices = lookup_indices.copy()
    for i, points_in_variable in list(enumerate(points_in_bins))[::-1]:
        print(points_in_variable)
        n_columns = len(points_in_variable)
        result[:, i] = points_in_variable[lookup_indices % n_columns]
        lookup_indices //= n_columns

    assert numpy.prod([len(x) for x in points_in_bins]) == len(lookup_indices)

    return result


def write_formula(inp_file, out_file, threshold):
    with open(inp_file) as formula_stream:
        features, bins_quantities, bins = unpack_formula(formula_stream, False)

    with open(inp_file) as formula_stream:
        mx = MatrixnetClassifier(formula_stream)

    bins_quantities = list(bins_quantities)
    for i in xrange(len(bins)):
        bins[i] = sorted(list(bins[i]))
        bins[i] = [-10 * abs(bins[i][0])] + bins[i]
        bins_quantities[i] += 1

    bins_quantities = numpy.array(bins_quantities)
    count = numpy.prod(bins_quantities)

    points_in_bins = []
    for i in range(len(features)):
        edges = numpy.array(bins[i])
        points_in = (edges[1:] + edges[:-1]) / 2.
        points_in = numpy.array(list(points_in) + [edges[-1] + 1.])
        points_in_bins.append(points_in)

    with open(out_file, "w") as output_stream:
        print "Total event count: " + str(count)

        output_stream.write(str(len(features)) + " # feature count\n")
        output_stream.write(" ".join([str(f) for f in features]) + " # features\n")
        output_stream.write(" ".join([str(b) for b in bins_quantities]) + "\n")
        for fbins in bins:
            output_stream.write(" ".join([str(b) for b in fbins]) + "\n")
            fbins.append(abs(fbins[-1]) * 3)

        divider = 10000
        output_stream.write(str(divider) + "\n")

        events = convert_lookup_index_to_bins(points_in_bins, lookup_indices=numpy.arange(count))
        predictions = expit(mx.apply(events))
        assert len(predictions) == count
        for q, pred in enumerate(predictions):
            if pred > threshold:
                output_stream.write(str(q) + " " + str(int(pred * divider)) + "\n")