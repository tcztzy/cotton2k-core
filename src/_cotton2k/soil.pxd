cdef extern from "Soil.h":
    ctypedef struct cSoilCell "SoilCell":
        double nitrate_nitrogen_content
        double fresh_organic_matter
        double water_content

    ctypedef struct SoilLayer:
        unsigned int number_of_left_columns_with_root
        unsigned int number_of_right_columns_with_root

    ctypedef struct cSoil "Soil":
        SoilLayer layers[40]
        cSoilCell cells[40][20]
