#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

template<class T>
struct NPIterator {
    pybind11::array_t<T>& data;

    T* begin() const {
        return data.mutable_data();
    }

    T* end() const {
        return data.mutable_data() + data.size();
    }
};

template<class T>
struct NPConstIterator {
    const pybind11::array_t<T>& data;

    const T* begin() const {
        return data.data();
    }

    const T* end() const {
        return data.data() + data.size();
    }
};

template<class Ta, class Tb>
struct NPZipIterator {
    class iterator {
    public:
        iterator(Ta* a, Tb* b): a(a), b(b) {}
        iterator operator++() { ++a; ++b; return *this; }
        bool operator!=(const iterator & other) const { return a != other.a; }
        const std::pair<Ta&, Tb&> operator*() const { return { *a, *b }; }
    private:
        Ta* a;
        Tb* b;
    };

    pybind11::array_t<Ta>& lhs;
    pybind11::array_t<Tb>& rhs;

    iterator begin() const {
        return iterator(lhs.mutable_data(), rhs.mutable_data());
    }

    iterator end() const {
        return iterator(lhs.mutable_data() + lhs.size(), rhs.mutable_data() + rhs.size());
    }
};

template<class Ta, class Tb>
struct NPConstZipIterator {
    class iterator {
    public:
        iterator(Ta* a, const Tb* b): a(a), b(b) {}
        iterator operator++() { ++a; ++b; return *this; }
        bool operator!=(const iterator & other) const { return a != other.a; }
        const std::pair<Ta&, const Tb&> operator*() const { return { *a, *b }; }
    private:
        Ta* a;
        const Tb* b;
    };

    pybind11::array_t<Ta>& lhs;
    const pybind11::array_t<Tb>& rhs;

    iterator begin() const {
        return iterator(lhs.mutable_data(), rhs.data());
    }

    iterator end() const {
        return iterator(lhs.mutable_data() + lhs.size(), rhs.data() + rhs.size());
    }
};

template<class T>
NPIterator<T> npIterate(pybind11::array_t<T>& data) {
    return { data };
}

template<class T>
NPConstIterator<T> npIterate(const pybind11::array_t<T>& data) {
    return { data };
}

template<class Ta, class Tb>
NPZipIterator<Ta, Tb> npZip(pybind11::array_t<Ta>& lhs, pybind11::array_t<Tb>& rhs) {
    return { lhs, rhs };
}

template<class Ta, class Tb>
NPConstZipIterator<Ta, Tb> npZip(pybind11::array_t<Ta>& lhs, const pybind11::array_t<Tb>& rhs) {
    return { lhs, rhs };
}

template<class T>
void npZero(pybind11::array_t<T>& data) {
    for (auto& x : npIterate(data)) x = 0;
}
