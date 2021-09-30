cdef extern from "Soil.h":
    ctypedef struct cSoilCell "SoilCell":
        double nitrate_nitrogen_content
        double fresh_organic_matter
        double water_content

    ctypedef struct cSoil "Soil":
        cSoilCell cells[40][20]
