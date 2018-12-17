#include <algorithm>
#include <vector>
#include <map>
#include <functional>

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

template<class T>
int indexOf(const std::vector<T>& arr, const T& item) {
	auto c = find(arr.begin(), arr.end(), item);
	assert(c != arr.end());
	return (int)(c - arr.begin());
}

template <class T, class V>
void sortByValueAscending (std::vector<T>& arr, std::function<V(const T&)> value) {
	std::sort(arr.begin(), arr.end(), [&](const T& a, const T& b) -> bool {
		return value(a) < value(b);
	});
}

template <class T, class V>
void sortByValueDescending (std::vector<T>& arr, std::function<V(const T&)> value) {
	std::sort(arr.begin(), arr.end(), [&](const T& a, const T& b) -> bool {
		return value(a) > value(b);
	});
}