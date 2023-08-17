#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

#include <omp.h>

#include <digestible/digestible.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "RunningStats.h"

namespace nb = nanobind;
using namespace nb::literals;

#define DTYPES float, double, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t

template<typename T>
constexpr const char *dtype_str()
{
    throw std::invalid_argument("Unsupported dtype");
    return "ERROR";
}

template<>
constexpr const char *dtype_str<float>()
{
    return "float32";
}
template<>
constexpr const char *dtype_str<double>()
{
    return "float64";
}
template<>
constexpr const char *dtype_str<int8_t>()
{
    return "int8";
}
template<>
constexpr const char *dtype_str<int16_t>()
{
    return "int16";
}
template<>
constexpr const char *dtype_str<int32_t>()
{
    return "int32";
}
template<>
constexpr const char *dtype_str<int64_t>()
{
    return "int64";
}
template<>
constexpr const char *dtype_str<uint8_t>()
{
    return "uint8";
}
template<>
constexpr const char *dtype_str<uint16_t>()
{
    return "uint16";
}
template<>
constexpr const char *dtype_str<uint32_t>()
{
    return "uint32";
}
template<>
constexpr const char *dtype_str<uint64_t>()
{
    return "uint64";
}

class OnlineStats
{
private:
    enum class dtypes { f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64 };

    template<typename Function>
    decltype(auto) dtype_switch(Function f) const
    {
        switch (m_dtype) {
        case dtypes::f32:
            return f.template operator()<float>();
            break;
        case dtypes::f64:
            return f.template operator()<double>();
            break;
        case dtypes::i8:
            return f.template operator()<int8_t>();
            break;
        case dtypes::i16:
            return f.template operator()<int16_t>();
            break;
        case dtypes::i32:
            return f.template operator()<int32_t>();
            break;
        case dtypes::i64:
            return f.template operator()<int64_t>();
            break;
        case dtypes::u8:
            return f.template operator()<uint8_t>();
            break;
        case dtypes::u16:
            return f.template operator()<uint16_t>();
            break;
        case dtypes::u32:
            return f.template operator()<uint32_t>();
            break;
        case dtypes::u64:
            return f.template operator()<uint64_t>();
            break;
        default:
            return f.template operator()<int8_t>(); // this should never happen, but we need it to silence a compiler warning
        }
    }

    template<typename Function>
    nb::ndarray<nb::numpy, double> stat_array(Function f) const
    {
        if (!m_n)
            throw nb::index_error("Cannot calculate statistic because no arrays were added.");
        double *buf = new double[m_size];
        nb::capsule owner(buf, [](void *p) noexcept {
            delete[] (double *)p;
        });

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            buf[i] = f(i);
        }
        return nb::ndarray<nb::numpy, double>(buf, m_ndim, m_shape, owner, nullptr, nb::dtype<double>());
    }

    template<typename Self, typename Function>
    auto digest_stat(Function f)
    {
        auto digests = (digestible::tdigest<Self> *)m_digests;
        return stat_array([digests, f](size_t i) {
            digests[i].merge();
            return f(digests[i]);
        });
    }

    template<typename Self>
    void init(const nb::ndarray<nb::device::cpu> &arr, size_t size)
    {
        m_size = arr.size();
        m_ndim = arr.ndim();
        if (m_ndim > 0) {
            m_shape = new size_t[m_ndim];
            m_stride = new size_t[m_ndim];
            m_stride_bytes = new size_t[m_ndim];
            for (size_t i = 0; i < m_ndim; ++i)
                m_shape[i] = arr.shape(i);
            size_t stride = 1;
            for (size_t i = m_ndim - 1; i > 0; --i) {
                m_stride[i] = stride;
                m_stride_bytes[i] = stride * sizeof(Self);
                stride *= m_shape[i];
            }
            m_stride[0] = stride;
            m_stride_bytes[0] = stride * sizeof(Self);
        }
        m_dtype = dtype<Self>();
        m_digests = operator new(m_size * sizeof(digestible::tdigest<Self>));
        auto digests = (digestible::tdigest<Self> *)m_digests;

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i)
            new (digests + i) digestible::tdigest<Self>(size);

        m_stats = new RunningStats[m_size];
    }

    template<typename Self, typename H, typename... Types>
    void select_add(const nb::ndarray<nb::device::cpu> &arr)
    {
        if (arr.dtype() == nb::dtype<H>())
            add_impl<Self, H>(arr);
        else if constexpr (sizeof...(Types) > 0)
            select_add<Self, Types...>(arr);
        else
            throw std::invalid_argument("Unsupported dtype");
    }

    template<typename Self, typename Dtype>
    void add_impl(const nb::ndarray<nb::device::cpu> &arr)
    {
        auto digests = (digestible::tdigest<Self> *)m_digests;
        const Dtype *data = (const Dtype *)arr.data();

        bool match_strides = true;
        for (size_t i = 0; i < m_ndim; ++i)
            if (arr.stride(i) != m_stride_bytes[i]) {
                match_strides = false;
                break;
            }

        if (match_strides) {
#pragma omp parallel for
            for (size_t i = 0; i < m_size; ++i) {
                digests[i].insert(data[i]);
                m_stats[i].push(data[i]);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < m_size; ++i) {
                auto idx = arridx(arr, i);
                digests[i].insert(data[idx]);
                m_stats[i].push(data[idx]);
            }
        }
        ++m_n;
    }

    inline size_t arridx(const nb::ndarray<nb::device::cpu> &arr, size_t i)
    {
        size_t idx = 0;
        for (size_t d = 0; d < m_ndim; ++d) {
            auto cdim = i / m_stride[d];
            idx += cdim * arr.stride(d);
            i -= cdim * m_stride[d];
        }
        if (i > 0)
            throw std::logic_error("i > 0 in arridx. This should never happen");
        return idx;
    }

    template<typename Dtype, typename... Ts>
    auto &get_data(const nb::ndarray<nb::c_contig, nb::device::cpu> &arr, size_t idx[], Ts... args)
    {
        if (sizeof...(args) < arr.ndim())
            return get_data<Dtype>(arr, idx, args..., idx[sizeof...(args)]);
        else
            return arr(args...);
    }

    template<typename H, typename... Types>
    void select_initial(const nb::ndarray<nb::device::cpu> &arr, size_t size)
    {
        if (arr.dtype() == nb::dtype<H>()) {
            init<H>(arr, size);
            select_add<H, DTYPES>(arr);
        } else if constexpr (sizeof...(Types) > 0)
            select_initial<Types...>(arr, size);
        else
            throw std::invalid_argument("Unsupported dtype");
    }

    template<typename H>
    constexpr dtypes dtype();

    void *m_digests = nullptr;
    RunningStats *m_stats = nullptr;
    size_t m_digestsize;
    size_t m_size;
    size_t m_ndim;
    size_t *m_shape = nullptr;
    size_t *m_stride = nullptr;
    size_t *m_stride_bytes = nullptr;
    dtypes m_dtype;
    uint64_t m_n = 0;

public:
    OnlineStats(const nb::ndarray<nb::device::cpu> &initial, size_t size = 20)
        : m_digestsize(size)
    {
        select_initial<DTYPES>(initial, size);
    }

    OnlineStats(size_t size = 20)
        : m_digestsize(size)
    {
    }

    ~OnlineStats()
    {
        if (m_digests != nullptr) {
            if (m_ndim > 0) {
                delete[] m_shape;
                delete[] m_stride;
                delete[] m_stride_bytes;
            }
            dtype_switch([this]<typename Self> {
                auto digests = (digestible::tdigest<Self> *)m_digests;

#pragma omp parallel for
                for (size_t i = 0; i < m_size; ++i) {
                    digests[i].~tdigest<Self>();
                }
            });
            operator delete(m_digests);
            delete[] m_stats;
        }
    }

    void reset()
    {
        m_n = 0;
        dtype_switch([this]<typename Self>() {
            auto digests = (digestible::tdigest<Self> *)m_digests;

#pragma omp parallel for
            for (size_t i = 0; i < m_size; ++i) {
                digests[i].reset();
                m_stats[i].clear();
            }
        });
    }

    void add(const nb::ndarray<nb::device::cpu> &arr)
    {
        if (m_digests == nullptr) {
            select_initial<DTYPES>(arr, m_digestsize);
            return;
        }

        auto msg = "Array shape does not match.";
        if (arr.ndim() != m_ndim)
            throw std::invalid_argument(msg);
        bool shape_matches = true;
        for (size_t i = 0; i < m_ndim; ++i)
            if (arr.shape(i) != m_shape[i]) {
                shape_matches = false;
                break;
            }
        if (!shape_matches)
            throw std::invalid_argument(msg);

        return dtype_switch([this, &arr]<typename Dtype> {
            return select_add<Dtype, DTYPES>(arr);
        });
    }

    auto quantile(double q)
    {
        if (q < 0 || q > 1)
            throw std::domain_error("Quantile must be between 0 and 1.");
        return dtype_switch([this, q]<typename Self> {
            return digest_stat<Self>([q](digestible::tdigest<Self> &digest) {
                return digest.quantile(100 * q);
            });
        });
    }

    auto max()
    {
        return dtype_switch([this]<typename Self> {
            return digest_stat<Self>([](digestible::tdigest<Self> &digest) {
                return digest.max();
            });
        });
    }

    auto min()
    {
        return dtype_switch([this]<typename Self> {
            return digest_stat<Self>([](digestible::tdigest<Self> &digest) {
                return digest.min();
            });
        });
    }

    template<typename Dtype>
    auto cdf(Dtype x)
    {
        return dtype_switch([this, x]<typename Self> {
            return digest_stat<Self>([x](digestible::tdigest<Self> &digest) {
                return digest.cumulative_distribution(x);
            });
        });
    }

    auto mean() const
    {
        return stat_array([this](size_t i) {
            return m_stats[i].mean();
        });
    }

    auto var() const
    {
        return stat_array([this](size_t i) {
            return m_stats[i].var();
        });
    }

    auto std() const
    {
        return stat_array([this](size_t i) {
            return m_stats[i].stddev();
        });
    }

    auto skewness() const
    {
        return stat_array([this](size_t i) {
            return m_stats[i].skewness();
        });
    }

    auto kurtosis() const
    {
        return stat_array([this](size_t i) {
            return m_stats[i].kurtosis();
        });
    }

    auto np_dtype() const
    {
        return nb::module_::import_("numpy").attr(dtype_switch([]<typename Self>() {
            return dtype_str<Self>();
        }));
    }

    auto np_shape() const
    {
        return nb::tuple(nb::cast(std::vector<size_t>(m_shape, m_shape + m_ndim)));
    }

    auto np_strides() const
    {
        return nb::tuple(nb::cast(std::vector<size_t>(m_stride_bytes, m_stride_bytes + m_ndim)));
    }

    size_t size() const
    {
        return m_size;
    }

    size_t ndim() const
    {
        return m_ndim;
    }

    uint64_t n_accumulated() const
    {
        return m_n;
    }

    std::string repr() const
    {
        std::string repr("NpOnlineStats object.");

        if (m_digests != nullptr) {
            repr += " Shape: (";
            if (m_ndim > 0) {
                for (size_t i = 0; i < m_ndim - 1; ++i)
                    repr += std::to_string(m_shape[i]) + ",";
                repr += std::to_string(m_shape[m_ndim - 1]);
            }
            repr += "), dtype: ";

            repr += dtype_switch([]<typename Self>() {
                return dtype_str<Self>();
            });
            repr += ".";
        }

        repr += " Accumulated " + std::to_string(m_n) + " array";
        if (m_n == 0 || m_n > 1)
            repr += "s";
        repr += ".";
        return repr;
    }
};

template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<float>()
{
    return dtypes::f32;
}
template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<double>()
{
    return dtypes::f64;
}
template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<int8_t>()
{
    return dtypes::i8;
}
template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<int16_t>()
{
    return dtypes::i16;
}
template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<int32_t>()
{
    return dtypes::i32;
}
template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<int64_t>()
{
    return dtypes::i64;
}
template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<uint8_t>()
{
    return dtypes::u8;
}
template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<uint16_t>()
{
    return dtypes::u16;
}
template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<uint32_t>()
{
    return dtypes::u32;
}
template<>
constexpr OnlineStats::dtypes OnlineStats::dtype<uint64_t>()
{
    return dtypes::u64;
}

NB_MODULE(_numpy_onlinestats_impl, m)
{
    nb::class_<OnlineStats>(m, "NpOnlineStats", R"___(
Streaming statistics for numpy arrays.

This class accumulates element-wise statistics for Numpy arrays in an online fashion:
The arrays themselves are not stored in memory. This enables calculation of (appriximate)
statistics for very large collections of arrays. Quantiles are approximated using the
t-digest algorithm and its `implementation <https://github.com/SpirentOrion/digestible>`_.
Moments are implemented using a `numerically stable algorithm <https://www.johndcook.com/blog/skewness_kurtosis/>`_.

References:
    Dunning and Ertl, 2019 (`arXiv:1902.04023 <http://arxiv.org/abs/1902.04023>`_)

Args:
    arr: First numpy array. Optional, if missing the accumulator will be initialized with the first call to `add`.
    size: Size of the t-digest buffer. Also used to determine the compression factor.

Note:
    All following arrays must have the same shape. The data type of
    the internal state is determined by `arr.dtype`, which may or may not be what you want.
    Pass `arr.astype(np.float64)` for best results.
)___")
        .def(nb::init<const nb::ndarray<nb::device::cpu> &, size_t>(), "arr"_a, "size"_a = 20)
        .def(nb::init<size_t>(), "size"_a = 20)
        .def_prop_ro("dtype", &OnlineStats::np_dtype, R"___(
The dtype of the accumulator.
)___")
        .def_prop_ro("shape", &OnlineStats::np_shape, R"___(
The shape of the accumulator.
)___")
        .def_prop_ro("strides", &OnlineStats::np_strides, R"___(
The strides of the accumulator.
)___")
        .def_prop_ro("size", &OnlineStats::size, R"___(
The size of the accumulator.
)___")
        .def_prop_ro("ndim", &OnlineStats::ndim, R"___(
The number of dimensions of the accumulator.
)___")
        .def_prop_ro("nacc", &OnlineStats::n_accumulated, R"___(
The number of samples accumulated.
)___")
        .def("add", &OnlineStats::add, "arr"_a, R"___(
Add an array to the accumulator.

Args:
    arr: An array.
)___")
        .def("reset", &OnlineStats::reset, R"___(
Reset the accumulator. Dtype and shape are kept, only the statistics are reset.
)___")
        .def("quantile", &OnlineStats::quantile, "q"_a, R"___(
Calculate an approximate quantile based on the current state.

Args:
    q: A quantile. Must be between 0 and 1.

Returns:
    A Numpy array.
)___")
        .def("max", &OnlineStats::max, R"___(
Get the element-wise maximum of all seen arrays.

Returns:
    A Numpy array.
)___")
        .def("min", &OnlineStats::min, R"___(
Get the element-wise minimum of all seen arrays.

Returns:
    A Numpy array.
)___")
        .def("cdf", &OnlineStats::cdf<float>, "x"_a, R"___(
Calculate the element-wise approximate cumulative distribution function.

Args:
    x: Value at which the CDF is to be calculated.

Returns: A Numpy array.
        )___")
        .def("cdf", &OnlineStats::cdf<double>)
        .def("cdf", &OnlineStats::cdf<int8_t>)
        .def("cdf", &OnlineStats::cdf<int16_t>)
        .def("cdf", &OnlineStats::cdf<int32_t>)
        .def("cdf", &OnlineStats::cdf<int64_t>)
        .def("cdf", &OnlineStats::cdf<uint8_t>)
        .def("cdf", &OnlineStats::cdf<uint16_t>)
        .def("cdf", &OnlineStats::cdf<uint32_t>)
        .def("cdf", &OnlineStats::cdf<uint64_t>)
        .def("mean", &OnlineStats::mean, R"___(
Calculate the element-wise mean of all seen arrays.

Returns: A Numpy array.
)___")
        .def("var", &OnlineStats::var, R"___(
Calculate the element-wise variance of all seen arrays.

Returns: A Numpy array.
)___")
        .def("std", &OnlineStats::std, R"___(
Calculate the element-wise standard deviation of all seen arrays.

Returns: A Numpy array.
)___")
        .def("skewness", &OnlineStats::skewness, R"___(
Calculate the element-wise skewness of all seen arrays.

Returns: A Numpy array.
)___")
        .def("kurtosis", &OnlineStats::kurtosis, R"___(
Calculate the element-wise kurtosis of all seen arrays.

Returns: A Numpy array.
)___")
        .def("__repr__", &OnlineStats::repr);
}
