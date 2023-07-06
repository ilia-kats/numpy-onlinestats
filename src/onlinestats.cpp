#include <algorithm>
#include <memory>

#include <omp.h>

#include <digestible/digestible.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <iostream>

namespace nb = nanobind;
using namespace nb::literals;

#define DTYPES float, double, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t

class OnlineStats
{
private:
    enum class dtypes { f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64 };

    template<typename Function>
    decltype(auto) dtype_switch(Function f)
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

    template<typename Self>
    void delete_digests()
    {
        auto digests = (digestible::tdigest<Self> *)m_digests;

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            digests[i].~tdigest<Self>();
        }
    }

    template<typename Self>
    nb::ndarray<nb::numpy, double> quantile_impl(double q)
    {
        auto digests = (digestible::tdigest<Self> *)m_digests;
        double *buf = new double[m_size];
        nb::capsule owner(buf, [](void *p) noexcept {
            delete[] (double *)p;
        });

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            digests[i].merge();
            buf[i] = digests[i].quantile(100 * q);
        }
        return nb::ndarray<nb::numpy, double>(buf, m_ndim, m_shape, owner, nullptr, nb::dtype<double>());
    }

    template<typename Self>
    void init(const nb::ndarray<nb::device::cpu> &arr)
    {
        m_size = arr.size();
        m_ndim = arr.ndim();
        m_shape = new size_t[m_ndim];
        m_stride = new size_t[m_ndim];
        for (size_t i = 0; i < m_ndim; ++i)
            m_shape[i] = arr.shape(i);
        size_t stride = 1;
        for (size_t i = m_ndim - 1; i > 0; --i) {
            m_stride[i] = stride;
            stride *= m_shape[i];
        }
        m_stride[0] = stride;
        m_dtype = dtype<Self>();
        m_digests = operator new(m_size * sizeof(digestible::tdigest<Self>));
        auto digests = (digestible::tdigest<Self> *)m_digests;

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i)
            new (digests + i) digestible::tdigest<Self>(20);
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

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            digests[i].insert(data[arridx(arr, i)]);
        }
    }

    inline size_t arridx(const nb::ndarray<nb::c_contig, nb::device::cpu> &arr, size_t i)
    {
        return i;
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
    void select_initial(const nb::ndarray<nb::device::cpu> &arr)
    {
        if (arr.dtype() == nb::dtype<H>()) {
            init<H>(arr);
            select_add<H, DTYPES>(arr);
        } else if constexpr (sizeof...(Types) > 0)
            select_initial<Types...>(arr);
        else
            throw std::invalid_argument("Unsupported dtype");
    }

    template<typename H>
    constexpr dtypes dtype();

    void *m_digests;
    size_t m_size;
    size_t m_ndim;
    size_t *m_shape;
    size_t *m_stride;
    dtypes m_dtype;

public:
    OnlineStats(const nb::ndarray<nb::device::cpu> &initial)
    {
        select_initial<DTYPES>(initial);
    }

    ~OnlineStats()
    {
        delete[] m_shape;
        delete[] m_stride;
        dtype_switch([this]<typename Dtype> {
            delete_digests<Dtype>();
        });
        operator delete(m_digests);
    }

    void add(const nb::ndarray<nb::device::cpu> &arr)
    {
        return dtype_switch([this, &arr]<typename Dtype> {
            return select_add<Dtype, DTYPES>(arr);
        });
    }

    nb::ndarray<nb::numpy, double> quantile(double q)
    {
        return dtype_switch([this, q]<typename Dtype> {
            return quantile_impl<Dtype>(q);
        });
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
    nb::class_<OnlineStats>(m, "NpOnlineStats")
        .def(nb::init<const nb::ndarray<nb::device::cpu> &>())
        .def("add", &OnlineStats::add)
        .def("quantile", &OnlineStats::quantile);
}
