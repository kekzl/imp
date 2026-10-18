#pragma once

// GraphExecutor by name only: headers that hold it by pointer include this, not executor.h
// (1442 lines, the most edited header in exec/), so an executor edit rebuilds its users only.
namespace imp {
class GraphExecutor;
}  // namespace imp
