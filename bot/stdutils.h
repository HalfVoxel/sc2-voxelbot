#include <algorithm>
#include <vector>

// -------------------------------------------------------------------
// --- Reversed iterable

template <typename T>
struct reversion_wrapper { T& iterable; };

template <typename T>
auto begin (reversion_wrapper<T> w) { return std::rbegin(w.iterable); }

template <typename T>
auto end (reversion_wrapper<T> w) { return std::rend(w.iterable); }

template <typename T>
reversion_wrapper<T> reverse (T&& iterable) { return { iterable }; }

template<class T>
bool contains(const std::vector<T>& arr, const T& item) {
	return find(arr.begin(), arr.end(), item) != arr.end();
}
