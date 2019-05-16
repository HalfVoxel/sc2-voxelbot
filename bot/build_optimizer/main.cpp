#include <libvoxelbot/utilities/mappings.h>
#include <libvoxelbot/buildorder/optimizer.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

int main() {
    initMappings();

    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        print(sys.path)
        sys.path.append("bot/python")
    )");

    // BuildOptimizer optimizer;
    // optimizer.init();
    // unitTestBuildOptimizer(optimizer);
    unitTestBuildOptimizer();
    return 0;
}